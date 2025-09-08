from typing import List, Tuple, Optional
import json
import gc
import re
import numpy as np

from Tool.Sentence_Segmenter import extract_sentences_spacy
from .semantic_common import (
    normalize_device,
    create_similarity_matrix,
    log_msg,
)

def semantic_grouping_main(
    passage_text: str,
    doc_id: str,
    embedding_model: str,
    *,
    # Graph clustering params
    knn_k: Optional[int] = None,
    edge_floor: float = 0.25,
    spectral_kmax: Optional[int] = None,
    # RMT + Modularity multiscale params
    rmt_keep_eigs: int = 3,
    mod_gamma_start: float = 0.7,
    mod_gamma_end: float = 1.6,
    mod_gamma_step: float = 0.15,
    # Post-processing params
    cap_soft: Optional[int] = None,
    small_group_min: int = 2,
    tau_merge: float = 0.38,
    reassign_delta: float = 0.02,
    # Graph clustering params
    # (kept above)
    embedding_batch_size: int = 64,
    device: Optional[str] = "cuda",
    silent: bool = False,
    collect_metadata: bool = False,
    sigmoid_tau_group: Optional[float] = None,
    engine: Optional[str] = None,
    **_extra,
) -> List[Tuple[str, str, Optional[str]]]:
    """Graph‑based Semantic Grouping – global semantic clusters from a similarity matrix (RMT + multiscale modularity; fallback k‑NN + Spectral),
        with split/merge post‑processing and a one‑pass boundary re‑assignment.

        Pipeline:
            1) Split into sentences; build similarity matrix S and optionally sharpen via sigmoid on z‑scores.
            2) Primary: Filter S with RMT (keep top eigen-components), then run multiscale modularity (python‑louvain) sweeping resolution γ.
               Fallback: Build symmetric weighted k‑NN graph from S and run Spectral clustering with automatic K via eigengap (K ≤ spectral_kmax).
            3) Post‑process: split over‑large clusters (internal k=2) when separable; merge undersized clusters if semantic gain is positive.
            5) One pass of boundary re‑assignment if moving a boundary sentence increases its score by at least reassign_delta.
            6) Emit clusters ordered by increasing sentence index.

        Key parameters:
            rmt_keep_eigs – number of top eigenvalues to keep in RMT filter (rest averaged as noise).
            mod_gamma_start/mod_gamma_end/mod_gamma_step – resolution sweep for multiscale modularity (larger γ → more clusters).
            knn_k, edge_floor, spectral_kmax – graph and K limit for Spectral fallback and internal split.
            cap_soft, small_group_min, tau_merge – disciplined split/merge controls.
            reassign_delta – improvement threshold to move a boundary sentence to another cluster.
        """
    # Start log (honor 'silent')
    log_msg(silent, f"[grouping] doc={doc_id} model={embedding_model}", 'info', 'grouping')

    # --- lightweight cleaning to remove residual metadata before sentence split ---
    def _preclean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        s = text
        # Remove lines/segments like: "Language: Spanish Article Type:BFN" (with or without quotes/spaces)
        s = re.sub(
            r"\s*[\"“”']{0,3}\s*Language:\s*\w+\s+Article\s*Type:\s*[A-Za-z0-9\-]+\.?\s*",
            " ",
            s,
            flags=re.IGNORECASE,
        )
        # Normalize excess whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    passage_text = _preclean_text(passage_text)
    sentences = extract_sentences_spacy(passage_text)
    if not sentences:
        return []

    if len(sentences) <= 1:
        # Too short – keep as a single chunk
        return [(f"{doc_id}_single", passage_text, None)]

    sim_matrix = create_similarity_matrix(
        sentences=sentences,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=normalize_device(device or 'cuda'),
        silent=silent,
    )
    if sim_matrix is None:
        return [(f"{doc_id}_matrix_fail", passage_text, None)]

    n = len(sentences)
    # Apply sigmoid to sharpen similarities around the global mean
    try:
        mu = float(np.mean(sim_matrix))
        sigma = float(np.std(sim_matrix) + 1e-9)
        tau = 0.15 if sigmoid_tau_group is None else float(sigmoid_tau_group)
        z = (sim_matrix - mu) / sigma
        sim_sharp = 1.0 / (1.0 + np.exp(-(z / tau)))
    except Exception:
        sim_sharp = sim_matrix
    # Remove self‑similarity
    try:
        np.fill_diagonal(sim_sharp, 0.0)
    except Exception:
        pass
    # Centrality for metadata/exemplar selection
    centrality = (sim_sharp.sum(axis=1) / max(n - 1, 1)).astype(float) if n > 1 else np.zeros(n, dtype=float)

    # ===== Helper functions =====
    def _mean_between(A: List[int], B: List[int]) -> float:
        if not A or not B:
            return 0.0
        return float(np.mean([float(sim_sharp[a, b]) for a in A for b in B]))

    def _mean_within(A: List[int]) -> float:
        if len(A) <= 1:
            return 1.0
        vals = []
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                vals.append(float(sim_sharp[A[i], A[j]]))
        return float(np.mean(vals)) if vals else 1.0

    # --- Random Matrix Theory (RMT) filtering to remove global/system noise ---
    def _rmt_filter(S: np.ndarray, keep_eigs: int = 3) -> np.ndarray:
        try:
            # Ensure symmetry
            S_sym = 0.5 * (S + S.T)
            # Eigen decomposition
            evals, evecs = np.linalg.eigh(S_sym)
            # Sort descending
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            k = int(max(1, min(keep_eigs, S.shape[0])))
            # Keep top-k, average the rest (noise level)
            if k < len(evals):
                noise_mean = float(np.mean(evals[k:]))
                evals_f = np.array([evals[i] if i < k else noise_mean for i in range(len(evals))], dtype=float)
            else:
                evals_f = evals.astype(float)
            S_f = (evecs @ np.diag(evals_f) @ evecs.T).astype(float)
            # Non-negativity and remove self-similarity
            S_f = np.maximum(S_f, 0.0)
            try:
                np.fill_diagonal(S_f, 0.0)
            except Exception:
                pass
            return S_f
        except Exception:
            # Fallback: return original (diagonal cleared)
            Sf = S.astype(float)
            try:
                np.fill_diagonal(Sf, 0.0)
            except Exception:
                pass
            return np.maximum(Sf, 0.0)

    # --- Multiscale modularity clustering over filtered similarity ---
    def _modularity_multiscale_labels(
        S_filtered: np.ndarray,
        gamma_start: float,
        gamma_end: float,
        gamma_step: float,
        edge_floor_local: float,
        kmax_cap: int,
    ) -> Optional[np.ndarray]:
        n_local = int(S_filtered.shape[0])
        if n_local <= 2:
            return None
        # Build adjacency by threshold
        A = np.where(S_filtered >= float(edge_floor_local), S_filtered, 0.0).astype(float)
        try:
            np.fill_diagonal(A, 0.0)
        except Exception:
            pass
        if np.allclose(A, 0.0):
            return None
        # Try python-louvain (community) via networkx
        best_labels = None
        best_k = 0
        best_score = float('inf')  # prefer fewer clusters; tie-break by cohesion
        try:
            import networkx as nx  # type: ignore
            import community as community_louvain  # type: ignore

            # Build graph from dense adjacency
            G = nx.Graph()
            G.add_nodes_from(range(n_local))
            for i in range(n_local):
                row = A[i]
                for j in range(i + 1, n_local):
                    w = float(row[j])
                    if w > 0.0:
                        G.add_edge(int(i), int(j), weight=w)

            if G.number_of_edges() == 0:
                return None

            # Sweep gamma, but build a consensus across multiple resolutions to increase stability.
            cur = float(gamma_start)
            gamma_end_eff = float(gamma_end)
            gamma_step_eff = float(gamma_step if gamma_step > 0 else 0.2)
            tried_any = False
            label_list = []
            gamma_list = []
            while cur <= gamma_end_eff + 1e-9:
                tried_any = True
                try:
                    part = community_louvain.best_partition(G, weight='weight', resolution=float(cur), random_state=0)
                    labels_arr = np.array([int(part.get(i, 0)) for i in range(n_local)], dtype=int)
                    k = int(np.max(labels_arr) + 1) if labels_arr.size else 0
                    if 2 <= k <= int(max(2, min(kmax_cap, n_local - 1))):
                        label_list.append(labels_arr)
                        gamma_list.append(float(cur))
                except Exception:
                    pass
                cur += gamma_step_eff

            if not label_list:
                return None

            # Consensus: compute pairwise co-association matrix and do spectral clustering on it.
            try:
                m = len(label_list)
                C = np.zeros((n_local, n_local), dtype=float)
                for lab in label_list:
                    for i in range(n_local):
                        for j in range(i + 1, n_local):
                            if lab[i] == lab[j]:
                                C[i, j] += 1.0
                                C[j, i] += 1.0
                C = C / float(m)
                # threshold weak co-associations
                thr = float(np.quantile(C[np.triu_indices(n_local, 1)], 0.5)) if n_local > 1 else 0.0
                Wc = np.where(C >= thr, C, 0.0)
                Wc = np.maximum(Wc, Wc.T)
                # choose K via eigengap on Laplacian of Wc
                if np.allclose(Wc, 0.0):
                    return label_list[-1]
                Lc = _normalized_laplacian(Wc)
                evals, evecs = np.linalg.eigh(Lc)
                order = np.argsort(evals)
                evals = evals[order]
                gaps = np.diff(evals[: min(len(evals)-1, kmax_cap) + 1])
                if gaps.size == 0:
                    k_final = 2
                else:
                    k_final = int(max(2, min(kmax_cap, int(np.argmax(gaps) + 1))))
                U = evecs[:, :k_final]
                row_norm = np.linalg.norm(U, axis=1) + 1e-9
                U_norm = U / row_norm[:, None]
                labels = _kmeans(U_norm, k=k_final, n_init=10, max_iter=200, seed=0)
                return labels
            except Exception:
                return label_list[-1]
        except Exception:
            # networkx/community not available → no modularity
            return None
        return None

    def _build_knn_graph(S: np.ndarray, kk: int, floor: float) -> np.ndarray:
        nn = S.shape[0]
        W = np.zeros((nn, nn), dtype=float)
        k_eff = int(max(1, min(kk, nn - 1)))
        for i in range(nn):
            row = S[i]
            idxs = np.argsort(-row)[: k_eff + 1]
            idxs = [int(j) for j in idxs if int(j) != i]
            for j in idxs:
                val = float(S[i, j])
                if val >= floor:
                    W[i, j] = val
        W = np.maximum(W, W.T)
        return W

    def _normalized_laplacian(W: np.ndarray) -> np.ndarray:
        d = np.sum(W, axis=1)
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        I = np.eye(W.shape[0], dtype=float)
        L = I - (D_inv_sqrt @ W @ D_inv_sqrt)
        return L

    def _kmeans(X: np.ndarray, k: int, n_init: int = 5, max_iter: int = 100, seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        best_labels = None
        best_inertia = float('inf')
        for _ in range(n_init):
            idx = rng.choice(X.shape[0], size=k, replace=False)
            centers = X[idx].copy()
            for _ in range(max_iter):
                dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                labels = np.argmin(dists, axis=1)
                new_centers = np.vstack([
                    X[labels == c].mean(axis=0) if np.any(labels == c) else centers[c]
                    for c in range(k)
                ])
                shift = float(np.linalg.norm(new_centers - centers))
                centers = new_centers
                if shift < 1e-6:
                    break
            inertia = float(((X - centers[labels]) ** 2).sum())
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
        return best_labels.astype(int)

    def _auto_k_spectral_labels(W: np.ndarray, kmax: int) -> Optional[np.ndarray]:
        nn = W.shape[0]
        if nn <= 2 or np.allclose(W, 0.0):
            return None
        L = _normalized_laplacian(W)
        try:
            evals, evecs = np.linalg.eigh(L)
        except Exception:
            return None
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        kmax_eff = int(max(2, min(kmax, nn - 1)))
        gaps = np.diff(evals[: kmax_eff + 1])
        if gaps.size == 0:
            k = 2
        else:
            k = int(np.argmax(gaps) + 1)
            k = max(2, min(k, kmax_eff))
        U = evecs[:, :k]
        row_norm = np.linalg.norm(U, axis=1) + 1e-9
        U_norm = U / row_norm[:, None]
        labels = _kmeans(U_norm, k=k, n_init=5, max_iter=100, seed=0)
        return labels

    # Auto-parameterization (no hard-coded values when possible)
    auto_params = bool(_extra.get('auto_params', True))
    # Effective K for kNN
    if auto_params:
        k_eff_all = int(max(5, min(32, round(n * 0.06))))
    else:
        k_eff_all = int(knn_k if knn_k is not None else max(5, min(20, n - 1)))
    # Effective edge floor by robust quantile on positive sims
    if auto_params:
        try:
            arr = np.asarray(sim_sharp, dtype=float)
            vals = arr[arr > 0.0]
            eff_edge_floor = float(np.quantile(vals, 0.80)) if vals.size else 0.4
        except Exception:
            eff_edge_floor = float(edge_floor)
    else:
        eff_edge_floor = float(edge_floor)
    W_all = _build_knn_graph(sim_sharp, kk=k_eff_all, floor=eff_edge_floor)

    # ===== Engine selection: 'spectral' only vs 'rmt' with fallback =====
    labels = None
    eng = (engine or "rmt").lower().strip()
    method_used = "RMT"
    if eng == "spectral":
        method_used = "SpectralOnly"
        if auto_params:
            kmax_eff = int(max(2, min(16, max(2, n // 6))))
        else:
            kmax_eff = int(spectral_kmax if spectral_kmax is not None else max(2, min(10, max(2, n // 5))))
        labels = _auto_k_spectral_labels(W_all, kmax=kmax_eff)
    else:
        try:
            S_f = _rmt_filter(sim_sharp, keep_eigs=int(max(1, rmt_keep_eigs)))
            labels = _modularity_multiscale_labels(
                S_filtered=S_f,
                gamma_start=float(mod_gamma_start),
                gamma_end=float(mod_gamma_end),
                gamma_step=float(mod_gamma_step),
                edge_floor_local=eff_edge_floor,
                kmax_cap=int((max(2, min(16, max(2, n // 6)))) if auto_params else (spectral_kmax if spectral_kmax is not None else max(2, min(10, max(2, n // 5)))))
            )
        except Exception:
            labels = None

        if labels is None:
            method_used = "SpectralFallback"
            if auto_params:
                kmax_eff = int(max(2, min(16, max(2, n // 6))))
            else:
                kmax_eff = int(spectral_kmax if spectral_kmax is not None else max(2, min(10, max(2, n // 5))))
            labels = _auto_k_spectral_labels(W_all, kmax=kmax_eff)

    if labels is None:
        groups: List[List[int]] = [list(range(n))]
    else:
        num_clusters = int(np.max(labels) + 1)
        groups = [[] for _ in range(num_clusters)]
        for i, lab in enumerate(labels.tolist()):
            groups[int(lab)].append(i)

    # ===== Post‑processing: split over‑large clusters with internal 2‑way spectral when separable =====
    # If cap_soft is not provided: derive from n
    if auto_params and cap_soft is None:
        eff_cap_soft = int(max(20, n // 4))
    else:
        eff_cap_soft = int(cap_soft if cap_soft is not None else max(20, n // 3))

    def _spectral_split_k2(members: List[int]) -> Optional[Tuple[List[int], List[int]]]:
        if len(members) < 4:
            return None
        subW = W_all[np.ix_(members, members)]
        L = _normalized_laplacian(subW)
        try:
            _, evecs = np.linalg.eigh(L)
        except Exception:
            return None
        U = evecs[:, :2]
        row_norm = np.linalg.norm(U, axis=1) + 1e-9
        U_norm = U / row_norm[:, None]
        lab2 = _kmeans(U_norm, k=2, n_init=5, max_iter=100, seed=1)
        left = [members[i] for i in range(len(members)) if lab2[i] == 0]
        right = [members[i] for i in range(len(members)) if lab2[i] == 1]
        if not left or not right:
            return None
        sep = _mean_between(left, right) - 0.5 * (_mean_within(left) + _mean_within(right))
        if sep < 0.0:
            return (sorted(left), sorted(right))
        return None

    new_groups: List[List[int]] = []
    for g in groups:
        if len(g) > eff_cap_soft:
            split = _spectral_split_k2(g)
            if split is not None and all(len(x) >= max(2, small_group_min) for x in split):
                new_groups.extend(list(split))
            else:
                new_groups.append(sorted(g))
        else:
            new_groups.append(sorted(g))
    groups = new_groups

    # ===== Conditionally merge undersized clusters =====
    merged: List[List[int]] = []
    consumed = set()
    # Auto small_group_min based on distribution of current group sizes
    if auto_params:
        try:
            sizes = [len(g) for g in groups]
            auto_min = int(max(2, np.percentile(sizes, 10))) if len(sizes) >= 5 else 2
        except Exception:
            auto_min = 2
        min_group_len_eff = auto_min
    else:
        min_group_len_eff = int(max(2, small_group_min))
    # Auto tau_merge: encourage merging only when inter > robust mid of sims
    if auto_params:
        try:
            arr = np.asarray(sim_sharp, dtype=float)
            vals = arr[arr > 0.0]
            eff_tau_merge = float(np.quantile(vals, 0.65)) if vals.size else float(tau_merge)
        except Exception:
            eff_tau_merge = float(tau_merge)
    else:
        eff_tau_merge = float(tau_merge)
    for i, g in enumerate(groups):
        if i in consumed:
            continue
        if len(g) >= max(2, int(min_group_len_eff)):
            merged.append(g)
            continue
        best_j = None
        best_gain = 0.0
        for j, h in enumerate(groups):
            if j == i or j in consumed:
                continue
            inter = _mean_between(g, h)
            if inter < float(eff_tau_merge):
                continue
            base = 0.5 * (_mean_within(g) + _mean_within(h))
            after = _mean_within(sorted(g + h))
            gain = after - base
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j is not None and best_gain > 0.0:
            consumed.add(best_j)
            merged.append(sorted(groups[best_j] + g))
        else:
            merged.append(g)

    # ===== One‑pass boundary sentence re‑assignment =====
    # ===== Refinement pass: adaptive split of loose clusters and merge near-duplicate neighbors =====
    try:
        # Compute internal means for all merged clusters
        internal_means = [float(_mean_within(g)) for g in merged]
        # Adaptive low threshold: clusters with internal mean below this are candidates to split
        if len(internal_means) >= 2:
            low_thr = float(np.percentile(np.array(internal_means, dtype=float), 25))
        else:
            low_thr = 0.0

        refined: List[List[int]] = []
        for idx, g in enumerate(merged):
            if len(g) >= 6 and float(_mean_within(g)) < max(0.5, low_thr):
                # Try spectral k=2 split if cluster is large but internally loose
                sp = _spectral_split_k2(g)
                if sp is not None:
                    left, right = sp
                    # Accept split only if both halves have higher internal coherence than parent
                    parent_mean = float(_mean_within(g))
                    left_mean = float(_mean_within(left))
                    right_mean = float(_mean_within(right))
                    if left_mean > parent_mean and right_mean > parent_mean:
                        refined.append(sorted(left))
                        refined.append(sorted(right))
                        continue
            refined.append(g)

        # Merge adjacent clusters when inter similarity is nearly as high as internal similarity
        merged_adj: List[List[int]] = []
        i = 0
        while i < len(refined):
            cur = refined[i]
            j = i + 1
            # attempt to merge with following clusters greedily when inter_mean >= 0.9 * min(internal_cur, internal_next)
            while j < len(refined):
                inter = _mean_between(cur, refined[j])
                cur_mean = float(_mean_within(cur))
                next_mean = float(_mean_within(refined[j]))
                cmp_thr = 0.9 * min(max(cur_mean, 1e-6), max(next_mean, 1e-6))
                # also allow merge if inter is above an absolute adaptive quantile of sim_sharp
                try:
                    all_vals = np.asarray(sim_sharp, dtype=float)
                    vals = all_vals[all_vals > 0.0]
                    global_merge_thr = float(np.quantile(vals, 0.60)) if vals.size else 0.5
                except Exception:
                    global_merge_thr = 0.5

                if inter >= max(cmp_thr, global_merge_thr):
                    # merge
                    cur = sorted(cur + refined[j])
                    j += 1
                else:
                    break
            merged_adj.append(cur)
            i = j

        merged = merged_adj
    except Exception:
        # On any error, keep original merged
        pass

    if len(merged) >= 2:
        # Auto reassignment improvement threshold
        if auto_params:
            try:
                arr = np.asarray(sim_sharp, dtype=float)
                vals = arr[arr > 0.0]
                eff_reassign_delta = float(np.std(vals)) * 0.1 if vals.size else float(reassign_delta)
            except Exception:
                eff_reassign_delta = float(reassign_delta)
        else:
            eff_reassign_delta = float(reassign_delta)
        for x in range(n):
            cur = None
            for cid, g in enumerate(merged):
                if x in g:
                    cur = cid
                    break
            if cur is None:
                continue
            cur_members = [y for y in merged[cur] if y != x]
            cur_mean = float(np.mean([float(sim_sharp[x, y]) for y in cur_members])) if cur_members else 0.0
            best_c = cur
            best_score = cur_mean
            for c2, h in enumerate(merged):
                if c2 == cur:
                    continue
                mean_other = float(np.mean([float(sim_sharp[x, y]) for y in h])) if h else 0.0
                # require a noticeable improvement
                if mean_other > best_score + float(eff_reassign_delta):
                    best_score = mean_other
                    best_c = c2
            if best_c != cur:
                merged[cur] = [y for y in merged[cur] if y != x]
                merged[best_c] = sorted(merged[best_c] + [x])

    final_chunks: List[Tuple[str, str, Optional[str]]] = []
    if collect_metadata:
        try:
            centrality_vals = list(map(float, centrality))
        except Exception:
            centrality_vals = []
        for i, g in enumerate(merged):
            chunk_sentences = [sentences[idx] for idx in sorted(set(g)) if 0 <= idx < n]
            if not chunk_sentences:
                continue
            chunk_text = " ".join(chunk_sentences).strip()
            if not chunk_text:
                continue
            # Exemplar = the member with the highest centrality within the cluster
            chunk_id_str = f"{doc_id}_cluster{i}"
            meta = {"chunk_id": chunk_id_str, "sent_indices": ",".join(str(x) for x in sorted(set(g))), "n": len(g), "method_used": method_used}
            if centrality_vals and g:
                try:
                    exemplar = max(g, key=lambda t: centrality_vals[t])
                    sims_ex = []
                    for idx2 in g:
                        if idx2 == exemplar:
                            continue
                        try:
                            sims_ex.append(float(sim_matrix[exemplar, idx2]))
                        except Exception:
                            continue
                    if sims_ex:
                        import math
                        m = sum(sims_ex) / len(sims_ex)
                        mn = min(sims_ex); mx = max(sims_ex)
                        var = sum((x - m) ** 2 for x in sims_ex) / len(sims_ex)
                        meta.update({
                            "exemplar": exemplar,
                            "sim_mean": round(m, 4),
                            "sim_min": round(mn, 4),
                            "sim_max": round(mx, 4),
                            "sim_std": round(math.sqrt(var), 4),
                            "exemplar_centrality": round(centrality_vals[exemplar], 4)
                        })
                except Exception:
                    pass
            final_chunks.append((chunk_id_str, chunk_text, json.dumps(meta, ensure_ascii=False)))
    else:
        for i, g in enumerate(merged):
            chunk_sentences = [sentences[idx] for idx in sorted(set(g)) if 0 <= idx < n]
            if not chunk_sentences:
                continue
            chunk_text = " ".join(chunk_sentences).strip()
            if not chunk_text:
                continue
            final_chunks.append((f"{doc_id}_cluster{i}", chunk_text, None))

    del sim_matrix
    gc.collect()

    if not final_chunks:
        return [(f"{doc_id}_fallback", passage_text, None)]

    # Completion log (honor 'silent')
    try:
        log_msg(silent, f"[grouping] doc={doc_id} method={method_used} clusters={len(final_chunks)}", 'info', 'grouping')
    except Exception:
        pass
    return final_chunks


def semantic_chunk_passage_from_grouping_logic(
    doc_id: str,
    passage_text: str,
    embedding_model: str = "thenlper/gte-base",
    *,
    # Graph clustering params
    knn_k: Optional[int] = None,
    edge_floor: float = 0.25,
    spectral_kmax: Optional[int] = None,
    # RMT + Modularity params
    rmt_keep_eigs: int = 3,
    mod_gamma_start: float = 0.7,
    mod_gamma_end: float = 1.6,
    mod_gamma_step: float = 0.15,
    # Post-processing params
    cap_soft: Optional[int] = None,
    small_group_min: int = 2,
    tau_merge: float = 0.38,
    reassign_delta: float = 0.02,
    embedding_batch_size: int = 64,
    device: Optional[str] = "cuda",
    silent: bool = False,
    collect_metadata: bool = False,
    sigmoid_tau_group: Optional[float] = None,
    engine: Optional[str] = None,
    **_extra,
) -> List[Tuple[str, str, Optional[str]]]:
    return semantic_grouping_main(
        passage_text=passage_text,
        doc_id=doc_id,
        embedding_model=embedding_model,
        cap_soft=cap_soft,
        small_group_min=small_group_min,
        tau_merge=tau_merge,
        knn_k=knn_k,
        edge_floor=edge_floor,
        spectral_kmax=spectral_kmax,
        rmt_keep_eigs=rmt_keep_eigs,
        mod_gamma_start=mod_gamma_start,
        mod_gamma_end=mod_gamma_end,
        mod_gamma_step=mod_gamma_step,
        reassign_delta=reassign_delta,
        embedding_batch_size=embedding_batch_size,
        device=device,
        silent=silent,
        collect_metadata=collect_metadata,
        sigmoid_tau_group=sigmoid_tau_group,
        engine=engine,
    )
