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

def _segment_centroid(embs: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    """Centroid of embs[a:b]; returns None if empty."""
    if b <= a:
        return None
    seg = embs[a:b]
    if seg.size == 0:
        return None
    c = seg.mean(axis=0)
    # Already L2-normalized rows; centroid may not be unit. Normalize to unit for cosine.
    nrm = float(np.linalg.norm(c) + 1e-9)
    return c / nrm

def _reassign_heads_tails(bounds: List[int], embs: np.ndarray, *, k: int = 2, margin: float = 0.02, min_first_boundary_index: int = 5) -> List[int]:
    """Post-pass local fix: move a few head sentences of the right segment to the left (or tail of left to right)
    if they are clearly more similar to the opposite centroid.

    - k: max sentences to reassign from head/tail per boundary.
    - margin: required similarity margin to trigger reassignment.
    """
    if not bounds:
        return bounds
    n = int(embs.shape[0])
    bnds = sorted(bounds)
    changed = False
    prev = 0
    for i, b in enumerate(bnds):
        nxt = bnds[i + 1] if (i + 1) < len(bnds) else n
        left_a, left_b = prev, b
        right_a, right_b = b, nxt
        # Ensure both sides have at least 1 sentence
        if left_b - left_a < 1 or right_b - right_a < 1:
            prev = b
            continue
        CL = _segment_centroid(embs, left_a, left_b)
        CR = _segment_centroid(embs, right_a, right_b)
        if CL is None or CR is None:
            prev = b
            continue
        # Try move head of right to left
        max_head = min(k, right_b - right_a - 1)  # keep at least 1 in right
        moved = 0
        for r in range(1, max_head + 1):
            head_c = _segment_centroid(embs, right_a, right_a + r)
            CR_wo = _segment_centroid(embs, right_a + r, right_b)
            if head_c is None or CR_wo is None:
                break
            scL = float(head_c @ CL)
            scR = float(head_c @ CR_wo)
            if scL > scR + float(margin):
                moved = r
            else:
                break
        if moved > 0:
            new_b = b + moved  # move head to left => boundary shifts right
            # Respect minimal indices
            min_nb = prev + 2
            if i == 0:
                min_nb = max(min_nb, int(min_first_boundary_index))
            if new_b < min_nb:
                new_b = min_nb
            if new_b < nxt - 1:  # keep at least 1 on the right
                bnds[i] = new_b
                b = new_b
                left_b = b
                right_a = b
                changed = True
                # recompute centroids after move
                CL = _segment_centroid(embs, left_a, left_b) or CL
                CR = _segment_centroid(embs, right_a, right_b) or CR
        # Try move tail of left to right
        max_tail = min(k, left_b - left_a - 1)
        moved = 0
        for r in range(1, max_tail + 1):
            tail_c = _segment_centroid(embs, left_b - r, left_b)
            CL_wo = _segment_centroid(embs, left_a, left_b - r)
            if tail_c is None or CL_wo is None:
                break
            scR = float(tail_c @ CR)
            scL = float(tail_c @ CL_wo)
            if scR > scL + float(margin):
                moved = r
            else:
                break
        if moved > 0:
            new_b = b - moved  # move tail to right => boundary shifts left
            min_nb = prev + 2
            if i == 0:
                min_nb = max(min_nb, int(min_first_boundary_index))
            if new_b < min_nb:
                new_b = min_nb
            if new_b >= right_a + 1:  # keep at least 1 on the left and right
                bnds[i] = new_b
                changed = True
        prev = bnds[i]
    return sorted(set(bnds))

def _dp_two_segment_best_cut(prefix_sum: np.ndarray, prefix_sq: np.ndarray, start: int, end: int, init_cut: int) -> Tuple[int, float]:
    """Given 1D signal s indexed at 0..m-1, we consider cuts between positions (start..end).
    Returns (best_cut, best_cost). Cost is SSE(left)+SSE(right) where SSE computed in O(1) using prefix sums.
    init_cut is the current cut for reference; it doesn't change optimization.
    Positions here align to 'between-sentences' indices for adj_sims (length n-1)."""
    def seg_cost(a: int, b: int) -> float:
        if b <= a:
            return 0.0
        s = prefix_sum[b] - prefix_sum[a]
        q = prefix_sq[b] - prefix_sq[a]
        l = b - a
        mean = s / float(l)
        # SSE = sum(x^2) - 2*mean*sum(x) + l*mean^2 = sumsq - l*mean^2
        return float(q - l * (mean * mean))
    best_j = init_cut
    best = float('inf')
    for j in range(start, end + 1):
        left = seg_cost(start, j)
        right = seg_cost(j, end + 1)
        c = left + right
        if c < best:
            best = c
            best_j = j
    return best_j, best

def _adjust_boundaries_by_local_dp(bounds: List[int], adj_signal: List[float], *, window: int = 6, improve_eps: float = 0.0, min_first_boundary_index: int = 5) -> List[int]:
    """Around each boundary, search best local cut within [b-window, b+window] minimizing two-segment SSE.
    This reduces sensitivity to min_boundary_spacing and refines positions using a DP criterion.
    """
    if not bounds:
        return bounds
    n = len(adj_signal) + 1  # number of sentences
    bs = sorted(bounds)
    # Build prefix sums for adj_signal
    s = np.array(adj_signal, dtype=float)
    prefix = np.concatenate([[0.0], np.cumsum(s)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(s * s)])
    new_bs: List[int] = []
    prev = 0
    for i, b in enumerate(bs):
        nxt = bs[i + 1] if (i + 1) < len(bs) else n
        lo = max(prev + 1, b - int(window))
        hi = min(nxt - 1, b + int(window))
        lo = max(lo, 1)
        hi = min(hi, n - 1)
        if lo > hi:
            new_bs.append(b)
            prev = b
            continue
        best_j, best_cost = _dp_two_segment_best_cut(prefix, prefix_sq, lo - 1, hi - 1, b - 1)
        new_b = best_j + 1
        # Enforce minimal position for the very first boundary
        if i == 0 and new_b < int(min_first_boundary_index):
            new_b = int(min_first_boundary_index)
        # Keep at least 1 sentence on both sides
        if new_b <= prev + 1:
            new_b = prev + 1
        if new_b >= nxt - 1:
            new_b = nxt - 1
        # Only accept if it changes and (optionally) improves cost compared to original cut
        if new_b != b:
            # compute original local cost
            _, orig_cost = _dp_two_segment_best_cut(prefix, prefix_sq, lo - 1, hi - 1, b - 1)
            if best_cost <= orig_cost - float(improve_eps):
                new_bs.append(new_b)
            else:
                new_bs.append(b)
        else:
            new_bs.append(b)
        prev = new_bs[-1]
    return sorted(set(new_bs))

def _global_dp_boundaries(adj_signal: List[float], *, penalty: float = 0.5, min_first_boundary_index: int = 1) -> List[int]:
    """Global O(n^2) DP on 1D signal to choose boundaries minimizing sum of in-segment SSE + penalty per cut.
    Returns sentence indices as boundaries (1..n-1). No hard length constraints besides at least 1 sentence per segment.
    """
    m = len(adj_signal)  # number of edges (n-1)
    if m <= 0:
        return []
    s = np.array(adj_signal, dtype=float)
    ps = np.concatenate([[0.0], np.cumsum(s)])
    ps2 = np.concatenate([[0.0], np.cumsum(s * s)])

    def seg_cost(a: int, b: int) -> float:
        # cost for s[a..b] inclusive
        if b < a:
            return 0.0
        S = ps[b + 1] - ps[a]
        Q = ps2[b + 1] - ps2[a]
        L = (b - a + 1)
        mu = S / float(L)
        return float(Q - L * (mu * mu))

    dp = [0.0] * m
    prev = [-1] * m
    for i in range(m):
        best = seg_cost(0, i)
        best_k = -1
        # consider last cut at k, new seg (k+1..i)
        for k in range(i):
            c = dp[k] + seg_cost(k + 1, i) + float(penalty)
            if c < best:
                best = c
                best_k = k
        dp[i] = best
        prev[i] = best_k

    # traceback
    cuts_edges: List[int] = []
    i = m - 1
    while i >= 0 and prev[i] != -1:
        k = prev[i]
        cuts_edges.append(k + 1)  # edge index
        i = k
    cuts_edges.sort()
    # convert edge indices to sentence boundaries (+1)
    boundaries = [e + 1 for e in cuts_edges]
    # enforce minimal first boundary if requested
    return [b for b in boundaries if b >= int(min_first_boundary_index)]

def _merge_short_segments(boundaries: List[int], embs: np.ndarray, *, short_len: int = 3) -> List[int]:
    """Merge segments shorter than short_len into the more similar neighbor based on centroid cosine.
    Iterate until no short segment remains or only one segment left.
    """
    n = int(embs.shape[0])
    if not boundaries:
        return boundaries
    b = sorted(boundaries)
    def centroid(a: int, c: int) -> Optional[np.ndarray]:
        return _segment_centroid(embs, a, c)
    changed = True
    while changed:
        changed = False
        segs = [(0, b[0])] + [(b[i], b[i + 1]) for i in range(len(b) - 1)] + [(b[-1], n)]
        if len(segs) <= 1:
            break
        for idx, (a, c) in enumerate(segs):
            L = c - a
            if L >= int(short_len):
                continue
            # decide direction to merge
            if idx == 0:
                # merge into right
                b.pop(0)  # remove first boundary
                changed = True
                break
            elif idx == len(segs) - 1:
                # merge into left
                b.pop(-1)
                changed = True
                break
            else:
                # choose neighbor with higher centroid similarity
                cl = centroid(segs[idx - 1][0], segs[idx - 1][1])
                cm = centroid(a, c)
                cr = centroid(segs[idx + 1][0], segs[idx + 1][1])
                sim_l = float(cm @ cl) if (cm is not None and cl is not None) else -1e9
                sim_r = float(cm @ cr) if (cm is not None and cr is not None) else -1e9
                if sim_l >= sim_r:
                    # merge into left => remove boundary before this short seg
                    # boundary index to remove is idx-1 (in boundaries list)
                    b.pop(idx - 1)
                else:
                    # merge into right => remove boundary at idx (current right boundary in list)
                    b.pop(idx)
                changed = True
                break
    return sorted(set(b))

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
    # Optional smoothing to reduce noise
    try:
        smooth_w = int(_legacy_kwargs.get('smooth_adj_window', 3) or 3)
    except Exception:
        smooth_w = 3
    adj_base = _median_smooth(adj_sims, window=smooth_w) if smooth_w and smooth_w > 1 else adj_sims
    adj_for_valley = adj_base
    try:
        sim_sigmoid_tau = _legacy_kwargs.get('sim_sigmoid_tau', None)
        if sim_sigmoid_tau is not None:
            tau_f = max(float(sim_sigmoid_tau), 1e-9)
            arr = np.array(adj_base, dtype=float)
            mu = float(arr.mean()); sd = float(arr.std() + 1e-9)
            z = (arr - mu) / sd
            adj_for_valley = (1.0 / (1.0 + np.exp(-(z / tau_f)))).tolist()
    except Exception:
        adj_for_valley = adj_base
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
    # Score map for NMS: prefer consensus
    if hybrid_mode == 'union_weighted':
        score_map = scores
    else:
        score_map = {}
        both = set(c99_bounds) & set(valley_bounds)
        for b in set(boundaries):
            if b in both:
                score_map[b] = 1.0
            elif b in set(valley_bounds):
                score_map[b] = 0.8
            else:
                score_map[b] = 0.7
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
        rmax = int(_legacy_kwargs.get('refine_max_shift', 1) or 1)
        boundaries = _refine_boundaries(boundaries, embeddings, max_shift=rmax, min_first_boundary_index=min_first_boundary_index)
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

    # 6b. Head–Tail reassignment to fix content drift near boundaries
    try:
        if boundaries:
            r_k = int(_legacy_kwargs.get('reassign_k', 2) or 2)
            r_margin = float(_legacy_kwargs.get('reassign_margin', 0.02) or 0.02)
            boundaries2 = _reassign_heads_tails(boundaries, embeddings, k=r_k, margin=r_margin, min_first_boundary_index=min_first_boundary_index)
            if boundaries2 != boundaries:
                boundaries = boundaries2
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

    # 6c. Local DP adjustment around boundaries on 1D signal for extra robustness (optional)
    try:
        if boundaries and bool(_legacy_kwargs.get('dp_local_adjust', True)):
            dpw = int(_legacy_kwargs.get('dp_window', 6) or 6)
            im_eps = float(_legacy_kwargs.get('dp_improve_eps', 0.0) or 0.0)
            boundaries2 = _adjust_boundaries_by_local_dp(boundaries, adj_base, window=dpw, improve_eps=im_eps, min_first_boundary_index=min_first_boundary_index)
            if boundaries2 != boundaries:
                boundaries = boundaries2
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

    # 8. Optional multi-pass: global DP segmentation on 1D signal, then merge short segments
    try:
        if bool(_legacy_kwargs.get('multi_pass_dp', False)):
            # Build a 1D signal; use smoothed base
            dp_penalty = float(_legacy_kwargs.get('dp_global_penalty', 0.6) or 0.6)
            dp_min_first = int(_legacy_kwargs.get('dp_min_first_boundary_index', min_first_boundary_index) or min_first_boundary_index)
            dp_bounds = _global_dp_boundaries(adj_base, penalty=dp_penalty, min_first_boundary_index=dp_min_first)
            # Union with current boundaries then soft-NMS to keep strong ones
            union_bs = sorted(set(boundaries) | set(dp_bounds))
            # simple score: +1 if from current, +1 if from DP
            sc = {b: 0.0 for b in union_bs}
            for b in union_bs:
                if b in boundaries:
                    sc[b] += 1.0
                if b in dp_bounds:
                    sc[b] += 1.0
            merged = _score_based_nms(union_bs, sc, min_boundary_spacing)
            # Optionally merge very short segments produced
            short_len = int(_legacy_kwargs.get('multi_pass_short_len', 3) or 3)
            if merged:
                merged2 = _merge_short_segments(merged, embeddings, short_len=short_len)
            else:
                merged2 = merged
            if merged2 != boundaries and merged2 is not None:
                boundaries = merged2
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
