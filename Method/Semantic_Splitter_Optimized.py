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

def _median_smooth(arr: List[float], window: int = 3) -> List[float]:
    """Simple median smoothing with odd window size. Fallbacks to original if window < 3."""
    w = int(window)
    if w <= 1:
        return list(arr)
    if w % 2 == 0:
        w += 1
    n = len(arr)
    if n == 0 or w <= 1 or w > max(1, n):
        return list(arr)
    half = w // 2
    padded = [arr[0]] * half + list(arr) + [arr[-1]] * half
    out = []
    for i in range(n):
        window_vals = padded[i:i + w]
        out.append(float(np.median(window_vals)))
    return out

def _score_based_nms(boundaries: List[int], score_of: dict, min_spacing: int) -> List[int]:
    """Greedy NMS that honors scores: keep higher-score boundaries when they are too close."""
    if not boundaries:
        return []
    spacing = max(1, int(min_spacing))
    # Sort by score desc, then by position asc for stability
    sorted_bs = sorted(boundaries, key=lambda b: (-float(score_of.get(b, 0.0)), int(b)))
    selected: List[int] = []
    for b in sorted_bs:
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
    # Optional smoothing to reduce noise
    # Auto parameter selection (no hard-coded thresholds) ---------------------
    auto_params = bool(_legacy_kwargs.get('auto_params', True))
    def _mad(x: np.ndarray) -> float:
        m = float(np.median(x)) if x.size else 0.0
        return float(np.median(np.abs(x - m)) + 1e-9)
    def _iqr(x: np.ndarray) -> float:
        return float(np.percentile(x, 75) - np.percentile(x, 25)) if x.size else 0.0

    try:
        smooth_w = int(_legacy_kwargs.get('smooth_adj_window', 3) or 3)
    except Exception:
        smooth_w = 3
    adj_base = _median_smooth(adj_sims, window=smooth_w) if smooth_w and smooth_w > 1 else adj_sims
    adj_for_valley = adj_base
    try:
        arr = np.array(adj_base, dtype=float)
        if auto_params:
            # Robust z using MAD, then sigmoid with tau from IQR
            med = float(np.median(arr))
            mad = _mad(arr)
            z = (arr - med) / (mad if mad > 0 else (float(arr.std()) + 1e-9))
            tau_auto = max(_iqr(arr) / 2.0, 0.05)
            adj_for_valley = (1.0 / (1.0 + np.exp(-(z / tau_auto)))).tolist()
        else:
            sim_sigmoid_tau = _legacy_kwargs.get('sim_sigmoid_tau', None)
            if sim_sigmoid_tau is not None:
                tau_f = max(float(sim_sigmoid_tau), 1e-9)
                mu = float(arr.mean()); sd = float(arr.std() + 1e-9)
                z = (arr - mu) / sd
                adj_for_valley = (1.0 / (1.0 + np.exp(-(z / tau_f)))).tolist()
    except Exception:
        adj_for_valley = adj_base
    # 4. Hybrid C99 + Valley (mặc định)
    # Dynamic spacing/first-index for long docs
    if auto_params:
        n = len(sentences)
        min_boundary_spacing = max(5, int(round(n / 50)))
        min_first_boundary_index = max(min_first_boundary_index, int(round(0.05 * n)))
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
    # Hybrid ensemble mode (prefer union_weighted for recall, then NMS)
    valley_tau = float(_legacy_kwargs.get('valley_tau', 0.12)) if not auto_params else max(_iqr(np.array(adj_base, dtype=float)) / 2.0, 0.06)
    hybrid_mode = str(_legacy_kwargs.get('hybrid_mode', 'intersection')).lower()
    if auto_params:
        hybrid_mode = 'union_weighted'
    valley_bounds: List[int] = _valley_boundaries(
        adj_for_valley,
        triplet_tau=valley_tau,
        min_boundary_spacing=min_boundary_spacing,
        min_first_boundary_index=min_first_boundary_index,
    )
    # Voting threshold derived from distribution (avoid magic numbers)
    if auto_params:
        vote_thr = 0.75
    else:
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
        score_map = scores
    elif hybrid_mode == 'union':
        boundaries = sorted(set(c99_bounds) | set(valley_bounds))
        # tạo score nhẹ để NMS
        score_map = {b: (1.0 if (b in c99_bounds and b in valley_bounds) else 0.8 if b in valley_bounds else 0.7) for b in boundaries}
    else:
        # intersection with preference to C99 positions when two methods are "near" each other
        try:
            tol = int(_legacy_kwargs.get('intersect_snap_tolerance', max(1, int(min_boundary_spacing) - 1)))
        except Exception:
            tol = max(1, int(min_boundary_spacing) - 1)
        cset = sorted(set(c99_bounds))
        vset = sorted(set(valley_bounds))
        chosen = []
        j = 0
        for c in cset:
            # advance valley pointer j to close to c
            while j < len(vset) and vset[j] < c - tol:
                j += 1
            ok = False
            if j < len(vset) and abs(vset[j] - c) <= tol:
                ok = True
            elif j > 0 and abs(vset[j - 1] - c) <= tol:
                ok = True
            if ok:
                chosen.append(c)
        boundaries = sorted(set(chosen))
        score_map = {b: 1.0 for b in boundaries}
    # Score-based NMS (soft preference)
    boundaries = _score_based_nms(boundaries, score_map, min_boundary_spacing)
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
    # 6. Hậu xử lý: refine/reassign/merge (auto; không cần tham số cứng)
    # 6a. Soft cap (tùy chọn): nếu một đoạn vượt quá cap câu, chèn điểm cắt gần vị trí cap theo cực tiểu adj-sim
    try:
        cap = _legacy_kwargs.get('soft_cap', None)
        cap = int(cap) if cap is not None else None
    except Exception:
        cap = None
    if auto_params and cap is None:
        # Soft cap dựa theo độ dài văn bản
        cap = max(24, int(round(n * 0.12)))
    if cap and int(cap) > 0:
        cap = int(cap)
        bs = sorted(list(boundaries))
        new_bs: List[int] = []
        prev = 0
        try:
            delta = int(_legacy_kwargs.get('soft_cap_delta', 2) or 2)
        except Exception:
            delta = 2
        for cut in bs + [n]:
            # chèn thêm cắt nhiều lần nếu đoạn quá dài
            while (cut - prev) > cap and (cut - prev) >= 3:
                target = prev + cap
                lo = max(prev + 1, target - delta)
                hi = min(cut - 1, target + delta)
                if hi <= lo:
                    break
                # local edges correspond to [lo-1 .. hi-1]
                l0 = max(prev, lo - 1)
                l1 = min(cut - 1, hi)
                local = np.array(adj_sims[l0:l1], dtype=float)
                if local.size <= 0:
                    break
                rel = int(np.argmin(local))
                pos = max(prev + 1, lo + rel)
                # đảm bảo ràng buộc biên đầu tiên
                if prev == 0 and pos < int(min_first_boundary_index):
                    pos = int(min_first_boundary_index)
                # đảm bảo còn ít nhất 1 câu mỗi phía
                if pos <= prev + 1:
                    pos = prev + 1
                if pos >= cut - 1:
                    pos = cut - 1
                new_bs.append(pos)
                prev = pos
            if cut != n:
                new_bs.append(cut)
            prev = cut
        if new_bs:
            boundaries = sorted(set(x for x in new_bs if 1 <= x < n))
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

    # 6b. Head–tail reassignment (1 pass): dịch biên tới cực tiểu adj-sim gần nhất trong cửa sổ nhỏ
    if auto_params and len(boundaries) > 0:
        win = 2
        try:
            arr = np.array(adj_base, dtype=float)
            new_b = []
            for b in sorted(boundaries):
                lo = max(1, b - win)
                hi = min(n - 1, b + win)
                if hi <= lo:
                    new_b.append(b)
                    continue
                local = arr[lo - 1:hi]
                if local.size == 0:
                    new_b.append(b)
                    continue
                nb = int(lo - 1 + np.argmin(local) + 1)
                nb = max(1, min(n - 1, nb))
                new_b.append(nb)
            boundaries = sorted(set(new_b))
            # rebuild chunks
            chunks = []
            groups = []
            cursor = 0
            for b in boundaries + [n]:
                grp = list(range(cursor, b))
                if grp:
                    chunks.append(" ".join(sentences[cursor:b]))
                    groups.append(grp)
                cursor = b
        except Exception:
            pass

    # 6c. Merge short segments theo auto min_len (char-based fallback nếu không có token)
    if auto_params and groups:
        lens = [len(g) for g in groups]
        min_len = max(3, int(round(np.percentile(lens, 10)))) if len(lens) >= 5 else 3
        merged_chunks: List[str] = []
        merged_groups: List[List[int]] = []
        buf_text = None
        buf_grp: List[int] = []
        for i, (ct, gp) in enumerate(zip(chunks, groups)):
            if buf_text is None:
                buf_text = ct; buf_grp = gp
            else:
                if len(buf_grp) < min_len:
                    buf_text = (buf_text + " " + ct).strip()
                    buf_grp = list(range(buf_grp[0], gp[-1] + 1))
                else:
                    merged_chunks.append(buf_text)
                    merged_groups.append(buf_grp)
                    buf_text = ct; buf_grp = gp
        if buf_text is not None:
            merged_chunks.append(buf_text)
            merged_groups.append(buf_grp)
        chunks, groups = merged_chunks, merged_groups

    # (removed) head–tail reassignment

    # (removed) local DP adjustment

    # (removed) length postprocess

    # (removed) multi-pass global DP
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
    chunks, sentences, groups = process_sentence_splitting_with_semantics(
        text=passage_text,
        embedding_model=embedding_model,
        device=device,
        min_boundary_spacing=min_boundary_spacing,
        min_first_boundary_index=min_first_boundary_index,
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
