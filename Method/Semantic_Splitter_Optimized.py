"""Semantic Splitter

Mục tiêu: Cắt văn bản thành các đoạn liên tiếp (contiguous) sao cho ranh giới rơi vào những điểm cắt ngữ nghĩa –
những chỗ độ tương đồng (cosine similarity) giữa hai câu liền kề giảm rõ rệt so với vùng xung quanh.

QUY TRÌNH 10 BƯỚC:
  1. Tách câu bằng spaCy (hoặc fallback) → danh sách sentences.
  2. Nhúng (embed) từng câu → vector đã chuẩn hoá L2.
  3. Tính similarity giữa MỌI cặp câu liền kề: sim(i,i+1) cho i=0..n-2.
  4. Với mỗi “edge” (giữa i và i+1) tính 4 tín hiệu boundary:
         • local_min: sim thấp hơn (hoặc bằng) 2 láng giềng (cực tiểu cục bộ).
         • low_z   : sim < mean_sim - k * std_sim (k = global_z_k) ⇒ thấp bất thường.
         • drop    : prev_sim - sim ≥ global_min_drop ⇒ rơi từ câu trước xuống hiện tại.
         • valley  : ((prev_sim + next_sim)/2 - sim) ≥ global_valley_prominence ⇒ sim “lọt hố” giữa hai bên cao hơn.
  5. Chọn edge là *candidate* nếu thỏa RULE:
            (is_local_min & có ≥1 tín hiệu phụ) HOẶC (không local_min nhưng đủ mạnh với ≥2 tín hiệu phụ).
  6. Cho mỗi candidate tính điểm ưu tiên:
            score = (1 - sim) + 0.7 * drop + 0.5 * valley
      (Ý tưởng: similarity càng thấp càng tốt; cộng thêm phần thưởng nếu rơi mạnh (drop) và thung lũng sâu (valley)).
  7. Sắp xếp candidates theo score giảm dần; duyệt lần lượt và CHẤP NHẬN boundary nếu không phá vỡ
      ràng buộc tối thiểu số câu mỗi đoạn (min_sentences_per_chunk) ở cả bên trái và bên phải.
  8. Nếu có target_sentences_per_chunk ⇒ nội suy số chunks mong muốn (desired_chunks). Nếu chọn chưa đủ boundary,
      dùng *uniform fallback* (chèn đều) nhưng vẫn tôn trọng min_sentences_per_chunk.
  9. (Tuỳ chọn) Nếu đặt max_sentences_per_chunk: phát hiện đoạn quá dài ⇒ ép chèn thêm boundary ở giữa (nếu hợp lệ).
 10. Gom nhóm câu theo danh sách boundary cuối cùng → trả về chuỗi chunk.

GHI CHÚ:
  • Thuật toán không điều chỉnh động thresholds – bạn điều khiển bằng 3 tham số: global_z_k, global_min_drop, global_valley_prominence.
  • target_sentences_per_chunk chỉ là “gợi ý mềm”: khi tín hiệu không đủ rõ sẽ chen đều.
  • Khi collect_metadata=True: tạo JSON thống kê nội bộ (similarity liên tiếp trong chunk).
"""

from typing import List, Optional, Tuple
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Tool.Sentence_Segmenter import extract_sentences_spacy
from .semantic_common import normalize_device, embed_sentences_batched, log_msg

def _embed(sentences: List[str], model_name: str, device, silent: bool) -> Optional[np.ndarray]:
    """Nhúng danh sách câu thành vector chuẩn hoá L2.

    Vì cosine_similarity(u,v) = u·v nếu ||u||=||v||=1, ta chuẩn hoá để:
      • Tăng ổn định số học.
      • Tránh phải tính cosine tường minh nhiều lần (dot product đủ dùng).
    """
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
    """Chọn danh sách chỉ số câu (boundary) sau khi đánh giá các tín hiệu.

    Parameters
    ----------
    adj_sims : list float
        similarity giữa câu i và i+1 (độ dài = n_sent-1)
    min_sentences : int
        Số câu tối thiểu mỗi chunk (ràng buộc cứng)
    target_sentences_per_chunk : Optional[int]
        Mục tiêu trung bình (gợi ý mềm) → suy ra desired_chunks
    global_z_k, global_min_drop, global_valley_prominence : float
        Ngưỡng cho 3 tín hiệu phụ
    global_max_extra : Optional[int]
        Giới hạn trên số boundary nếu muốn “cắt tối đa X lần”
    max_sentences_per_chunk : Optional[int]
        Giới hạn cứng kích thước chunk (nếu vượt ép thêm boundary)
    """
    n_edges = len(adj_sims)
    if n_edges == 0:
        return []
    sims_arr = np.array(adj_sims, dtype=float)
    mean_sim = float(np.mean(sims_arr))
    std_sim = float(np.std(sims_arr) + 1e-9)
    candidates = []  # (boundary, score)
    # --- (B1) Quét từng edge để tính tín hiệu ---
    for i, sim in enumerate(adj_sims):  # edge giữa (i, i+1)
        boundary = i + 1
        prev_sim = adj_sims[i-1] if i > 0 else sim
        next_sim = adj_sims[i+1] if i+1 < n_edges else sim
        # drop_prev: mức rơi từ similarity trước sang hiện tại
        drop_prev = prev_sim - sim if i > 0 else 0.0
        # valley_prom: độ sâu “thung lũng” so với trung bình hai bên
        valley_prom = ((prev_sim + next_sim)/2.0) - sim
        low_z = sim < mean_sim - global_z_k * std_sim
        triggers = int(low_z) + int(drop_prev >= global_min_drop) + int(valley_prom >= global_valley_prominence)
        # local_min: thấp hơn (hoặc bằng) hai láng giềng – edge tốt để cắt
        is_local_min = (sim <= prev_sim) and (sim <= next_sim) and ((i==0) or (i==n_edges-1) or (sim < prev_sim or sim < next_sim))
        # RULE hợp lệ
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
    # --- (B2) Sắp xếp & chọn greedily với ràng buộc kích thước tối thiểu ---
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
    # --- (B3) Uniform fallback nếu chưa đạt số boundary kỳ vọng ---
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
    # --- (B4) Ép cắt nếu chunk quá dài ---
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
    # 1. Tách câu
    sentences = extract_sentences_spacy(text)
    if not sentences:
        return [], [], []
    if len(sentences) <= min_sentences_per_chunk:
        # Văn bản quá ngắn – giữ nguyên
        return [" ".join(sentences)], sentences, [list(range(len(sentences)))]
    # 2. Nhúng câu
    device_r = normalize_device(device)
    embeddings = _embed(sentences, embedding_model, device_r, silent)
    if embeddings is None:
        return [" ".join(sentences)], sentences, [list(range(len(sentences)))]
    # 3. Similarity liền kề (adjacent)
    adj_sims = [float(cosine_similarity(embeddings[i].reshape(1,-1), embeddings[i+1].reshape(1,-1))[0,0])
                for i in range(len(sentences)-1)]
    # 4. Chọn boundary
    boundaries = _select_boundaries(adj_sims, min_sentences_per_chunk, target_sentences_per_chunk,
                                    global_z_k, global_min_drop, global_valley_prominence,
                                    global_max_extra, max_sentences_per_chunk)
    n = len(sentences)
    all_bounds = boundaries + [n]
    chunks: List[str] = []
    groups: List[List[int]] = []
    cursor = 0
    # 5. Gom chunk dựa trên boundary
    for b in all_bounds:
        grp = list(range(cursor, b))
        if grp:
            if len(grp) < min_sentences_per_chunk and groups:
                # Tránh chunk lẻ tẻ quá ngắn: nhập vào chunk trước
                groups[-1].extend(grp)
                chunks[-1] = " ".join(sentences[groups[-1][0]: groups[-1][-1]+1])
            else:
                chunks.append(" ".join(sentences[cursor:b]))
                groups.append(grp)
        cursor = b
    # 6. Hợp nhất lần cuối nếu vẫn còn nhóm nhỏ (< min)
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
    collect_metadata: bool = False,
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
    collect_metadata: bool = False,
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
        collect_metadata=collect_metadata,
        **legacy_kwargs,
    )
