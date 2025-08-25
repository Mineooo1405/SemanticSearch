from typing import List, Optional, Tuple
import json
import numpy as np
import re
from Tool.Sentence_Segmenter import extract_sentences_spacy
from .semantic_common import normalize_device, embed_sentences_batched, log_msg

def _embed(sentences: List[str], model_name: str, device, silent: bool) -> Optional[np.ndarray]:
    """Embed a list of sentences into L2‑normalized vectors.

    Note: with unit vectors, dot product equals cosine; improves numerical stability.
    """
    if not sentences:
        return None
    embs = embed_sentences_batched(sentences, model_name, base_batch_size=32, device=device, silent=silent)
    if embs is None or embs.size == 0:
        return None
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embs / norms


def _c99_boundaries(embs: np.ndarray,
                    min_chunk_size: int = 3,
                    max_cuts: Optional[int] = None,
                    min_gain: float = 0.01,
                    *,
                    use_local_rank: bool = False,
                    mask_size: int = 11,
                    stopping: str = "gain",
                    knee_c: float = 1.2,
                    smooth_window: int = 3) -> List[int]:
    n = embs.shape[0]
    if n < 2 * int(min_chunk_size):
        return []
    # 1) Similarity matrix (L2 normalized -> dot product is cosine)
    S = embs @ embs.T
    # 2) Rank matrix
    if use_local_rank:
        # Local mask ranking (close to C99). Cost O(n^2 * mask^2), suited for medium n.
        m = max(3, int(mask_size) | 1)  # force odd >= 3
        half = m // 2
        R = np.zeros_like(S, dtype=np.float32)
        for i in range(n):
            i0 = 0 if i - half < 0 else i - half
            i1 = n if i + half + 1 > n else i + half + 1
            for j in range(n):
                j0 = 0 if j - half < 0 else j - half
                j1 = n if j + half + 1 > n else j + half + 1
                win = S[i0:i1, j0:j1]
                v = S[i, j]
                denom = win.size if win.size > 0 else 1
                # fraction of entries in the window lower than current value
                R[i, j] = float((win < v).sum()) / float(denom)
    else:
        # Global row/column ranking (fast variant)
        row_less = (S[:, None, :] < S[:, :, None]).sum(axis=2).astype(np.int32)
        col_less_T = (S.T[:, None, :] < S.T[:, :, None]).sum(axis=2).astype(np.int32)
        col_less = col_less_T.T
        R = (row_less + col_less).astype(np.float32)

    def _inside_density(rank_mat: np.ndarray, segments: List[Tuple[int, int]]) -> float:
        total_sum = 0.0
        total_area = 0
        for a, b in segments:
            if b <= a:
                continue
            block = rank_mat[a:b, a:b]
            total_sum += float(block.sum()) if block.size > 0 else 0.0
            ln = (b - a)
            total_area += ln * ln
        return total_sum / float(total_area) if total_area > 0 else 0.0
    # 3) Divisive clustering on rank matrix
    segs: List[Tuple[int, int]] = [(0, n)]
    cuts: List[int] = []
    D_series: List[float] = [_inside_density(R, segs)]
    while True:
        best_gain, best_pos, best_idx = -1e9, None, None
        best_mean_all = 0.0
        for idx, (a, b) in enumerate(segs):
            if (b - a) < 2 * int(min_chunk_size):
                continue
            r_sub = R[a:b, a:b]
            mean_all = float(r_sub.mean()) if r_sub.size > 0 else 0.0
            for c in range(a + int(min_chunk_size), b - int(min_chunk_size) + 1):
                left_sub = R[a:c, a:c]
                right_sub = R[c:b, c:b]
                left_mean = float(left_sub.mean()) if left_sub.size > 0 else mean_all
                right_mean = float(right_sub.mean()) if right_sub.size > 0 else mean_all
                gain = 0.5 * (left_mean + right_mean) - mean_all
                if gain > best_gain:
                    best_gain, best_pos, best_idx = gain, c, idx
                    best_mean_all = mean_all
        # adaptive gain threshold – avoid under/over-segmentation
        adaptive_thr = max(float(min_gain), 0.1 * abs(best_mean_all))
        if best_pos is None or (max_cuts is not None and len(cuts) >= int(max_cuts)):
            break
        # If stopping by 'gain', apply threshold immediately; for 'profile', keep collecting stats.
        if stopping.lower() == "gain" and best_gain < adaptive_thr:
            break
        a, b = segs.pop(int(best_idx))
        segs += [(a, best_pos), (best_pos, b)]
        cuts.append(int(best_pos))
        D_series.append(_inside_density(R, sorted(segs)))
        if stopping.lower() == "gain" and best_gain < adaptive_thr:
            break
    if stopping.lower() != "profile" or not cuts:
        return sorted(set(cuts))
    # Profile-based stopping D(n): knee on δD (moving average)
    deltas = np.diff(np.array(D_series, dtype=float))
    if deltas.size == 0:
        return sorted(set(cuts))
    sw = max(1, int(smooth_window))
    if sw > 1 and deltas.size >= sw:
        kernel = np.ones(sw, dtype=float) / float(sw)
        dsm = np.convolve(deltas, kernel, mode='same')
    else:
        dsm = deltas
    mu = float(dsm.mean())
    sd = float(dsm.std() + 1e-9)
    thr = mu - float(knee_c) * sd
    # choose m at the first large drop in δD (dsm < thr)
    knee_idx = None
    for i, v in enumerate(dsm, start=1):  # i == number of segments after i-th split (i.e., m)
        if v < thr:
            knee_idx = i
            break
    if knee_idx is None:
        return sorted(set(cuts))
    m = max(1, int(knee_idx))
    keep = min(m - 1, len(cuts))
    return sorted(set(cuts[:keep]))


def _valley_boundaries(
    adj_sims: List[float],
    *,
    triplet_tau: float = 0.12,
    min_boundary_spacing: int = 2,
    min_first_boundary_index: int = 5,
) -> List[int]:
    """Valley detection based on adjacent similarity + sigmoid.

    - Detect local valleys: decreasing then increasing run; take the minimum index in the run.
    - Strength = (sim[i-1]-sim[i])_+ + (sim[i+1]-sim[i])_+ at the valley.
    - Z-normalize strength, then apply sigmoid with temperature `triplet_tau`.
    - Filter by `min_first_boundary_index` and apply NMS with `min_boundary_spacing`.
    """
    n = len(adj_sims)
    if n < 3:
        return []

    sims = np.array(adj_sims, dtype=float)

    # 1) Find valley candidates by decreasing→increasing runs, use the minimum index in the run
    raw_valleys: List[Tuple[int, float]] = []  # (boundary_index, raw_strength)
    i = 1
    while i <= n - 2:
        if not (sims[i] <= sims[i - 1]):
            i += 1
            continue
        j = i
        min_idx = i
        min_val = sims[i]
        while j + 1 <= n - 2 and sims[j + 1] <= sims[j]:
            j += 1
            if sims[j] < min_val:
                min_val = sims[j]
                min_idx = j
        # confirm there is an increasing phase afterwards
        if j < n - 1 and sims[j + 1] >= sims[j]:
            left_drop = max(0.0, float(sims[min_idx - 1] - sims[min_idx])) if min_idx > 0 else 0.0
            right_rise = max(0.0, float(sims[min_idx + 1] - sims[min_idx])) if (min_idx + 1) < n else 0.0
            strength = left_drop + right_rise
            raw_valleys.append((min_idx + 1, strength))  # boundary sits between min_idx and min_idx+1
        i = j + 1

    if not raw_valleys:
        return []

    # 2) Z-normalize and sigmoid-transform strength
    strengths = np.array([s for (_b, s) in raw_valleys], dtype=float)
    mu = float(strengths.mean())
    sd = float(strengths.std() + 1e-9)
    z = (strengths - mu) / sd
    tau = max(float(triplet_tau), 1e-9)
    scores = 1.0 / (1.0 + np.exp(-(z / tau)))

    # 3) Filter by first-boundary constraint and NMS by score
    cands: List[Tuple[int, float, float]] = []  # (b, score, raw)
    for (b, s), sc in zip(raw_valleys, scores):
        if b < int(min_first_boundary_index):
            continue
        cands.append((int(b), float(sc), float(s)))

    if not cands:
        return []

    cands.sort(key=lambda x: (-x[1], -x[2]))
    selected: List[int] = []
    spacing = max(1, int(min_boundary_spacing))
    for b, sc, s in cands:
        if all(abs(b - x) >= spacing for x in selected):
            selected.append(b)

    return sorted(set(selected))

def process_sentence_splitting_with_semantics(
    text: str,
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    min_boundary_spacing: int = 5,
    min_first_boundary_index: int = 5,
    # post-process length control (non-hard):
    min_chunk_len: Optional[int] = None,
    max_chunk_len: Optional[int] = None,
    silent: bool = True,
    **_legacy_kwargs,
) -> Tuple[List[str], List[str], List[List[int]]]:
    # 0. Lightweight pre-cleaning: strip frequent header metadata and normalize whitespace
    def _preclean_text(t: str) -> str:
        if not isinstance(t, str):
            return ""
        s = t
        s = re.sub(r'^Language:\s*\w+\s+Article\s*Type:\s*[^\s\[\]]*\s*(?:\[Text\])?\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*["“”\']{0,3}\s*Language:\s*\w+\s+Article\s*Type:\s*[A-Za-z0-9\-]+\.?' , ' ', s, flags=re.IGNORECASE)
        s = re.sub(r'\[Article by[^\]]*\]\s*', '', s)
        s = re.sub(r'\[Report by[^\]]*\]\s*', '', s)
        s = re.sub(r'\[From the[^\]]*\]\s*', '', s)
        s = re.sub(r'\[Excerpts?\]\s*', '', s)
        s = re.sub(r'\[Text\]\s*', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    text = _preclean_text(text)
    # 1. Sentence segmentation
    sentences = extract_sentences_spacy(text)
    if not sentences:
        return [], [], []
    if len(sentences) <= 1:
        # Document too short – keep as is
        return [" ".join(sentences)], sentences, [list(range(len(sentences)))]
    # 2. Sentence embeddings
    device_r = normalize_device(device)
    embeddings = _embed(sentences, embedding_model, device_r, silent)
    if embeddings is None:
        return [" ".join(sentences)], sentences, [list(range(len(sentences)))]
    # 3. Adjacent similarity (and windowed features if needed)
    n_sent = len(sentences)
    # Always compute full adjacent similarities for post-processing (C99 does not need windows)
    adj_sims = [float(embeddings[i] @ embeddings[i + 1]) for i in range(n_sent - 1)]
    # Optional: z-score + sigmoid transform for the valley detector
    adj_for_valley = adj_sims
    try:
        sim_sigmoid_tau = _legacy_kwargs.get('sim_sigmoid_tau', None)
        if sim_sigmoid_tau is not None:
            tau_f = max(float(sim_sigmoid_tau), 1e-9)
            arr = np.array(adj_sims, dtype=float)
            mu = float(arr.mean()); sd = float(arr.std() + 1e-9)
            z = (arr - mu) / sd
            adj_for_valley = (1.0 / (1.0 + np.exp(-(z / tau_f)))).tolist()
    except Exception:
        adj_for_valley = adj_sims
    # 4. Hybrid C99 + Valley
    c99_min_chunk = max(3, int(min_boundary_spacing))
    c99_bounds: List[int] = _c99_boundaries(
        embeddings,
        min_chunk_size=c99_min_chunk,
        max_cuts=None,
        use_local_rank=bool(_legacy_kwargs.get('c99_use_local_rank', False)),
        mask_size=int(_legacy_kwargs.get('c99_mask_size', 11) or 11),
        stopping=str(_legacy_kwargs.get('c99_stopping', 'gain')), # gain or profile
        knee_c=float(_legacy_kwargs.get('c99_knee_c', 1.2) or 1.2),
        smooth_window=int(_legacy_kwargs.get('c99_smooth_window', 3) or 3),
    )
    # Valley from adjacent similarities
    valley_tau = float(_legacy_kwargs.get('valley_tau', 0.12))
    hybrid_mode = str(_legacy_kwargs.get('hybrid_mode', 'intersection')).lower()
    valley_bounds: List[int] = _valley_boundaries(
        adj_for_valley,
        triplet_tau=valley_tau,
        min_boundary_spacing=min_boundary_spacing,
        min_first_boundary_index=min_first_boundary_index,
    )
    vote_thr = float(_legacy_kwargs.get('vote_thr', 0.8) or 0.8)
    if hybrid_mode == 'union_weighted':
        all_bs = sorted(set(c99_bounds) | set(valley_bounds))
        scores = {}
        for b in all_bs:
            sc = 0.0
            if b in valley_bounds:
                sc += 0.5
            if b in c99_bounds:
                sc += 0.5
            scores[b] = sc
        boundaries = [b for b in all_bs if scores.get(b, 0.0) >= vote_thr]
    elif hybrid_mode == 'union':
        boundaries = sorted(set(c99_bounds) | set(valley_bounds))
    else:
        boundaries = sorted(set(c99_bounds) & set(valley_bounds))
    # NMS with spacing
    nms_bounds: List[int] = []
    for b in boundaries:
        if not nms_bounds or abs(b - nms_bounds[-1]) >= int(min_boundary_spacing):
            nms_bounds.append(b)
    boundaries = nms_bounds
    # Fallback: when intersection is empty use C99 (avoid over-cutting from naive union)
    if hybrid_mode == 'intersection' and len(boundaries) == 0:
        boundaries = c99_bounds
    n = len(sentences)
    all_bounds = boundaries + [n]
    chunks: List[str] = []
    groups: List[List[int]] = []
    cursor = 0
    # 5. Build chunks from boundaries
    for b in all_bounds:
        grp = list(range(cursor, b))
        if grp:
            chunks.append(" ".join(sentences[cursor:b]))
            groups.append(grp)
        cursor = b
    # 6. Fine-tune boundaries: local shift near boundary to minimize cross-similarity
    def _refine_boundaries(bounds: List[int], embs: np.ndarray, max_shift: int = 0, min_first_boundary_index: int = 3) -> List[int]:
        if not bounds:
            return bounds
        n = embs.shape[0]
        refined: List[int] = []
        prev = 0
        sorted_bounds = sorted(bounds)
        for idx, b in enumerate(sorted_bounds):
            next_b = sorted_bounds[idx + 1] if (idx + 1) < len(sorted_bounds) else n
            best_b = b
            best_score = -1e9
            for delta in range(-max_shift, max_shift + 1):
                nb = b + delta
                # Ensure at least 1 sentence on each side after shifting
                min_nb_allowed = prev + 2
                if idx == 0:
                    min_nb_allowed = max(min_nb_allowed, int(min_first_boundary_index))
                if nb < min_nb_allowed or nb >= next_b - 1:
                    continue
                left = embs[prev:nb]
                right = embs[nb:next_b]
                if left.shape[0] < 1 or right.shape[0] < 1:
                    continue
                a = embs[nb - 1]
                c = embs[nb]
                cross = float(a @ c)
                # lower cross-similarity is better
                score = -cross
                if score > best_score:
                    best_score = score
                    best_b = nb
            refined.append(best_b)
            prev = best_b
        return sorted(set(refined))

    try:
        # reuse embeddings already normalized
        boundaries = _refine_boundaries(boundaries, embeddings, max_shift=1, min_first_boundary_index=min_first_boundary_index)
        all_bounds = boundaries + [n]
        chunks = []
        groups = []
        cursor = 0
        for b in all_bounds:
            grp = list(range(cursor, b))
            if grp:
                chunks.append(" ".join(sentences[cursor:b]))
                groups.append(grp)
            cursor = b
    except Exception:
        pass

    # 7. Length post-process (optional): merge-too-short, split-too-long
    def _postprocess_by_length(bounds: List[int], min_len: Optional[int], max_len: Optional[int], sent_count: int, adj_sims_full: List[float]) -> List[int]:
        if not bounds:
            return bounds
        if (min_len is None and max_len is None) or sent_count <= 2:
            return bounds
        b = sorted(bounds)
        # Merge too-short (including last segment)
        if min_len is not None and min_len > 1:
            merged: List[int] = []
            prev = 0
            for cut in b + [sent_count]:
                seg_len = cut - prev
                if seg_len >= min_len:
                    if cut != sent_count:
                        merged.append(cut)
                    prev = cut
            # If last segment still < min_len and we have >=2 boundaries, drop the last cut
            if merged and (sent_count - merged[-1]) < min_len and len(merged) > 1:
                merged.pop()
            b = merged
        # Split too-long: insert multiple cuts at weakest local adj-sim minima
        if max_len is not None and max_len > 1:
            refined: List[int] = []
            prev = 0
            arr = np.array(adj_sims_full, dtype=float)
            for cut in b + [sent_count]:
                seg_end = cut
                while (seg_end - prev) > max_len and (seg_end - prev) >= 3:
                    start = max(prev, 0)
                    end = min(seg_end, len(arr))
                    if end <= start:
                        break
                    local = arr[start:end]
                    pos = int(np.argmin(local)) + start + 1
                    refined.append(pos)
                    prev = pos
                refined.append(seg_end)
                prev = seg_end
            b = sorted(set(x for x in refined if 1 <= x < sent_count))
        return sorted(set(b))

    try:
        new_bounds = _postprocess_by_length(boundaries, min_chunk_len, max_chunk_len, n, adj_sims)
        if new_bounds != boundaries:
            boundaries = new_bounds
            all_bounds = boundaries + [n]
            chunks = []
            groups = []
            cursor = 0
            for b in all_bounds:
                grp = list(range(cursor, b))
                if grp:
                    chunks.append(" ".join(sentences[cursor:b]))
                    groups.append(grp)
                cursor = b
    except Exception:
        pass
    return chunks, sentences, groups

def semantic_splitter_main(
    doc_id: str,
    passage_text: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    min_boundary_spacing: int = 2,
    min_first_boundary_index: int = 5,
    silent: bool = False,
    collect_metadata: bool = False,
    **legacy_kwargs,
) -> List[Tuple[str, str, Optional[str]]]:
    _min_len = legacy_kwargs.pop('min_chunk_len', None)
    _max_len = legacy_kwargs.pop('max_chunk_len', None)

    chunks, sentences, groups = process_sentence_splitting_with_semantics(
        text=passage_text,
        embedding_model=embedding_model,
        device=device,
        min_boundary_spacing=min_boundary_spacing,
        min_first_boundary_index=min_first_boundary_index,
        min_chunk_len=_min_len,
        max_chunk_len=_max_len,
        silent=silent,
        **legacy_kwargs,
    )
    if not chunks:
        sentences = extract_sentences_spacy(passage_text)
        return [(f"{doc_id}_fallback", passage_text, None)] if sentences else []
    out: List[Tuple[str, str, Optional[str]]] = []
    if collect_metadata:
        # Re-embed sentences only if we did not already (we lost embeddings scope here) – simplest is to recompute for metadata
        try:
            device_r = normalize_device(device)
            sentences = extract_sentences_spacy(passage_text)
            embs = _embed(sentences, embedding_model, device_r, True) if sentences else None
        except Exception:
            embs = None
        for idx, grp in enumerate(groups):
            ctext = " ".join(sentences[grp[0]: grp[-1]+1]) if sentences and grp else ""
            if not ctext:
                continue
            # Removed hash generation per user request
            chunk_id_str = f"{doc_id}_chunk{idx}"
            meta = {"chunk_id": chunk_id_str, "sent_indices": ",".join(str(i) for i in grp), "n": len(grp)}
            if embs is not None and len(grp) > 1:
                # adjacency similarities inside group (contiguous)
                sims = []
                for a, b in zip(grp, grp[1:]):
                    try:
                        va, vb = embs[a], embs[b]
                        sims.append(float((va @ vb)))
                    except Exception:
                        continue
                if sims:
                    import math
                    m = sum(sims)/len(sims)
                    mn = min(sims); mx = max(sims)
                    var = sum((x-m)**2 for x in sims)/len(sims)
                    meta.update({"sim_mean": round(m,4), "sim_min": round(mn,4), "sim_max": round(mx,4), "sim_std": round(math.sqrt(var),4)})
            out.append((chunk_id_str, ctext, json.dumps(meta, ensure_ascii=False)))
    else:
        for idx, ctext in enumerate(chunks):
            out.append((f"{doc_id}_chunk{idx}", ctext, None))
    return out

def chunk_passage_text_splitter(
    doc_id: str,
    passage_text: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    min_boundary_spacing: int = 2,
    min_first_boundary_index: int = 3,
    silent: bool = False,
    collect_metadata: bool = False,
    **legacy_kwargs,
) -> List[Tuple[str, str, Optional[str]]]:
    return semantic_splitter_main(
        doc_id=doc_id,
        passage_text=passage_text,
        embedding_model=embedding_model,
        device=device,
        min_boundary_spacing=min_boundary_spacing,
        min_first_boundary_index=min_first_boundary_index,
        silent=silent,
        collect_metadata=collect_metadata,
        **legacy_kwargs,
    )
