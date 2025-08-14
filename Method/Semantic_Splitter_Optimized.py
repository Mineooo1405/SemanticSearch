"""Semantic Splitter (Minimal Global Semantic Breakpoint Version)

Chỉ giữ đúng 1 thuật toán: chọn điểm cắt toàn cục từ độ tương đồng cosine giữa các câu liền kề.

Tín hiệu boundary (giữa câu i và i+1):
  - local_min: similarity là cực tiểu cục bộ
  - low_z: similarity < mean - k*std
  - drop: prev_sim - sim >= global_min_drop
  - valley: ((prev_sim + next_sim)/2 - sim) >= global_valley_prominence

Rule chọn candidate:
  (local_min và >=1 tín hiệu khác) HOẶC (không local_min nhưng >=2 tín hiệu).

Sau đó chấm điểm score = (1 - sim) + 0.7*drop + 0.5*valley và chọn greedily với ràng buộc độ dài tối thiểu.

Chấp nhận mọi tham số legacy qua **kwargs nhưng bỏ qua (đảm bảo controller cũ không lỗi).
Trả về danh sách (chunk_id, chunk_text, None).
"""

from typing import List, Optional, Tuple
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Tool.Sentence_Segmenter import extract_sentences_spacy
from .semantic_common import normalize_device, embed_sentences_batched, log_msg

def _embed(sentences: List[str], model_name: str, device, silent: bool) -> Optional[np.ndarray]:
    if not sentences:
        return None
    embs = embed_sentences_batched(sentences, model_name, base_batch_size=32, device=device, silent=silent)
    if embs is None or embs.size == 0:
        return None
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embs / norms

def _select_boundaries(adj_sims: List[float], min_sentences: int, target_sentences_per_chunk: Optional[int],
                       global_z_k: float, global_min_drop: float, global_valley_prominence: float,
                       global_max_extra: Optional[int], max_sentences_per_chunk: Optional[int]) -> List[int]:
    n_edges = len(adj_sims)
    if n_edges == 0:
        return []
    sims_arr = np.array(adj_sims, dtype=float)
    mean_sim = float(np.mean(sims_arr))
    std_sim = float(np.std(sims_arr) + 1e-9)
    candidates = []  # (boundary, score)
    for i, sim in enumerate(adj_sims):
        boundary = i + 1
        prev_sim = adj_sims[i-1] if i > 0 else sim
        next_sim = adj_sims[i+1] if i+1 < n_edges else sim
        drop_prev = prev_sim - sim if i > 0 else 0.0
        valley_prom = ((prev_sim + next_sim)/2.0) - sim
        low_z = sim < mean_sim - global_z_k * std_sim
        triggers = int(low_z) + int(drop_prev >= global_min_drop) + int(valley_prom >= global_valley_prominence)
        is_local_min = (sim <= prev_sim) and (sim <= next_sim) and ((i==0) or (i==n_edges-1) or (sim < prev_sim or sim < next_sim))
        include = (is_local_min and triggers >= 1) or ((not is_local_min) and triggers >= 2)
        if include:
            score = (1 - sim) + 0.7*max(drop_prev,0) + 0.5*max(valley_prom,0)
            candidates.append((boundary, score))

    n_sent = n_edges + 1
    desired_chunks = None
    if target_sentences_per_chunk and target_sentences_per_chunk > 0:
        desired_chunks = max(1, round(n_sent / target_sentences_per_chunk))
        while desired_chunks > 1 and n_sent < desired_chunks * min_sentences:
            desired_chunks -= 1
    max_boundaries = (desired_chunks - 1) if desired_chunks else None
    if global_max_extra is not None:
        max_boundaries = global_max_extra if max_boundaries is None else min(max_boundaries, global_max_extra)

    selected: List[int] = []
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        for b, score in candidates:
            last = selected[-1] if selected else 0
            if b - last < min_sentences:  # enforce left segment size
                continue
            if (n_sent - b) < min_sentences:  # enforce right remainder size
                continue
            selected.append(b)
            if max_boundaries is not None and len(selected) >= max_boundaries:
                break

    # Uniform fallback if target unmet
    if desired_chunks and len(selected) < (desired_chunks - 1):
        step = max(min_sentences, int(round(n_sent / desired_chunks)))
        pos = step
        uniform = []
        while pos < n_sent and len(uniform) < (desired_chunks - 1):
            if n_sent - pos >= min_sentences:
                uniform.append(pos)
            pos += step
        merged = sorted(set(selected + uniform))
        filtered = []
        prev = 0
        for b in merged:
            if b - prev < min_sentences or n_sent - b < min_sentences:
                continue
            filtered.append(b)
            prev = b
        selected = filtered

    # Enforce maximum sentences per chunk if requested
    if max_sentences_per_chunk:
        while True:
            b_sorted = sorted(selected)
            starts = [0] + b_sorted
            ends = b_sorted + [n_sent]
            oversized = None
            for s, e in zip(starts, ends):
                if e - s > max_sentences_per_chunk:
                    oversized = (s, e)
                    break
            if oversized is None:
                break
            s, e = oversized
            center = s + (e - s)//2
            if center not in selected and (center - s) >= min_sentences and (e - center) >= min_sentences:
                selected.append(center)
            else:
                break
    return sorted(selected)

def process_sentence_splitting_with_semantics(
    text: str,
    *,
    min_sentences_per_chunk: int = 6,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    target_sentences_per_chunk: Optional[int] = None,
    global_z_k: float = 0.50,
    global_min_drop: float = 0.12,
    global_valley_prominence: float = 0.04,
    global_max_extra: Optional[int] = None,
    max_sentences_per_chunk: Optional[int] = None,
    silent: bool = True,
    **_legacy_kwargs,
) -> Tuple[List[str], List[str], List[List[int]]]:
    sentences = extract_sentences_spacy(text)
    if not sentences:
        return [], [], []
    if len(sentences) <= min_sentences_per_chunk:
        return [" ".join(sentences)], sentences, [list(range(len(sentences)))]
    device_r = normalize_device(device)
    embeddings = _embed(sentences, embedding_model, device_r, silent)
    if embeddings is None:
        return [" ".join(sentences)], sentences, [list(range(len(sentences)))]
    adj_sims = [float(cosine_similarity(embeddings[i].reshape(1,-1), embeddings[i+1].reshape(1,-1))[0,0]) for i in range(len(sentences)-1)]
    boundaries = _select_boundaries(adj_sims, min_sentences_per_chunk, target_sentences_per_chunk,
                                    global_z_k, global_min_drop, global_valley_prominence,
                                    global_max_extra, max_sentences_per_chunk)
    n = len(sentences)
    all_bounds = boundaries + [n]
    chunks: List[str] = []
    groups: List[List[int]] = []
    cursor = 0
    for b in all_bounds:
        grp = list(range(cursor, b))
        if grp:
            if len(grp) < min_sentences_per_chunk and groups:
                groups[-1].extend(grp)
                chunks[-1] = " ".join(sentences[groups[-1][0]: groups[-1][-1]+1])
            else:
                chunks.append(" ".join(sentences[cursor:b]))
                groups.append(grp)
        cursor = b
    # Defensive merge pass
    if any(len(g) < min_sentences_per_chunk for g in groups) and len(groups) > 1:
        merged_chunks: List[str] = []
        merged_groups: List[List[int]] = []
        i = 0
        while i < len(groups):
            g = groups[i]
            if len(g) >= min_sentences_per_chunk:
                merged_groups.append(g)
                merged_chunks.append(" ".join(sentences[g[0]: g[-1]+1]))
                i += 1
            else:
                if i+1 < len(groups):
                    new_g = g + groups[i+1]
                    merged_groups.append(new_g)
                    merged_chunks.append(" ".join(sentences[new_g[0]: new_g[-1]+1]))
                    i += 2
                elif merged_groups:
                    merged_groups[-1].extend(g)
                    merged_chunks[-1] = " ".join(sentences[merged_groups[-1][0]: merged_groups[-1][-1]+1])
                    i += 1
                else:
                    merged_groups.append(g)
                    merged_chunks.append(" ".join(sentences[g[0]: g[-1]+1]))
                    i += 1
        groups = merged_groups
        chunks = merged_chunks
    return chunks, sentences, groups

def semantic_splitter_main(
    doc_id: str,
    passage_text: str,
    min_sentences_per_chunk: int = 6,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    target_sentences_per_chunk: Optional[int] = None,
    global_z_k: float = 0.50,
    global_min_drop: float = 0.12,
    global_valley_prominence: float = 0.04,
    global_max_extra: Optional[int] = None,
    max_sentences_per_chunk: Optional[int] = None,
    silent: bool = False,
    include_oie: bool = False,
    **legacy_kwargs,
) -> List[Tuple[str, str, Optional[str]]]:
    if include_oie and not silent:
        log_msg(silent, "OIE disabled (legacy ignored)", 'debug', 'split')
    chunks, sentences, groups = process_sentence_splitting_with_semantics(
        text=passage_text,
        min_sentences_per_chunk=min_sentences_per_chunk,
        embedding_model=embedding_model,
        device=device,
        target_sentences_per_chunk=target_sentences_per_chunk,
        global_z_k=global_z_k,
        global_min_drop=global_min_drop,
        global_valley_prominence=global_valley_prominence,
        global_max_extra=global_max_extra,
        max_sentences_per_chunk=max_sentences_per_chunk,
        silent=silent,
        **legacy_kwargs,
    )
    if not chunks:
        sentences = extract_sentences_spacy(passage_text)
        return [(f"{doc_id}_fallback", passage_text, None)] if sentences else []
    out: List[Tuple[str, str, Optional[str]]] = []
    for idx, ctext in enumerate(chunks):
        h = hashlib.sha1(ctext.encode('utf-8', errors='ignore')).hexdigest()[:8]
        out.append((f"{doc_id}_chunk{idx}_hash{h}", ctext, None))
    return out

def chunk_passage_text_splitter(
    doc_id: str,
    passage_text: str,
    min_sentences_per_chunk: int = 6,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    target_sentences_per_chunk: Optional[int] = None,
    global_z_k: float = 0.50,
    global_min_drop: float = 0.12,
    global_valley_prominence: float = 0.04,
    global_max_extra: Optional[int] = None,
    max_sentences_per_chunk: Optional[int] = None,
    silent: bool = False,
    **legacy_kwargs,
) -> List[Tuple[str, str, Optional[str]]]:
    return semantic_splitter_main(
        doc_id=doc_id,
        passage_text=passage_text,
        min_sentences_per_chunk=min_sentences_per_chunk,
        embedding_model=embedding_model,
        device=device,
        target_sentences_per_chunk=target_sentences_per_chunk,
        global_z_k=global_z_k,
        global_min_drop=global_min_drop,
        global_valley_prominence=global_valley_prominence,
        global_max_extra=global_max_extra,
        max_sentences_per_chunk=max_sentences_per_chunk,
        silent=silent,
        **legacy_kwargs,
    )
