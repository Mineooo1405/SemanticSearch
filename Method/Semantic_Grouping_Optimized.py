from typing import List, Tuple, Optional
import json
import gc
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
    min_sentences_per_chunk: int = 6,
    anchor_target_sentences: int = 6,
    anchor_similarity_floor: float = 0.0,
    embedding_batch_size: int = 64,
    device: Optional[str] = "cuda",
    silent: bool = False,
    collect_metadata: bool = False,
    **_extra,
) -> List[Tuple[str, str, Optional[str]]]:
    """Anchor Greedy Grouping – Tạo các cụm câu không liên tiếp (non‑contiguous) dựa trên group sentence.

        Quy trình 8 BƯỚC:
            1. Tách câu bằng spaCy (hoặc fallback) → danh sách sentences.
            2. Trường hợp ngắn: nếu tổng số câu < min_sentences_per_chunk ⇒ giữ nguyên thành 1 chunk.
            3. Nếu số câu ≤ anchor_target_sentences ⇒ cũng trả về 1 chunk (không cần tách).
            4. Tính ma trận similarity đầy đủ (n x n) giữa các câu.
            5. Tính centrality cho từng câu: trung bình similarity tới các câu khác ⇒ đo “độ đại diện”.
            6. Vòng lặp greedily:
                     • Chọn anchor = câu có centrality lớn nhất trong tập còn lại.
                     • Lọc những câu còn lại có similarity ≥ anchor_similarity_floor.
                     • Sort giảm dần theo similarity tới anchor, lấy top (anchor_target_sentences - 1) ⇒ tạo nhóm.
                     • Loại các câu vừa chọn khỏi tập remaining.
            7. Xử lý nhóm nhỏ (undersized) bằng cách merge/attach để đảm bảo mọi chunk cuối ≥ min_sentences_per_chunk.
            8. Kết hợp câu trong mỗi nhóm (theo thứ tự chỉ số tăng dần) → chunk_text.

        GHI CHÚ QUAN TRỌNG:
            • Các câu trong chunk có thể ở vị trí rời rạc — không giữ tính liên tục.
            • Overlap similarity ranges giữa các chunk là bình thường vì mỗi chunk dùng anchor khác để tính thống kê.
            • anchor_similarity_floor giúp bỏ các câu quá xa anchor, tránh nhiễu.
            • collect_metadata=True: lưu JSON gồm chỉ số câu, thống kê similarity anchor→members và centrality anchor.

        Parameters:
                min_sentences_per_chunk: Số câu tối thiểu mỗi chunk sau cùng.
                anchor_target_sentences: Kích thước mong muốn mỗi nhóm (anchor + các câu chọn thêm).
                anchor_similarity_floor: Ngưỡng tối thiểu để một câu được gắn vào anchor.
                embedding_batch_size: Batch size embedding.
                device: Thiết bị.
                silent: Ẩn log.
        """
    if not silent:
        log_msg(False, f"[anchor_greedy] doc={doc_id} model={embedding_model}", 'info', 'group')

    sentences = extract_sentences_spacy(passage_text)
    if not sentences:
        return []

    if len(sentences) < min_sentences_per_chunk:
        # No hash: simple deterministic ID
        return [(f"{doc_id}_short", passage_text, None)]

    target_k = max(anchor_target_sentences, min_sentences_per_chunk)
    if len(sentences) <= target_k:
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
    centrality = ((sim_matrix.sum(axis=1) - 1.0) / (n - 1)).astype(float) if n > 1 else np.zeros(n, dtype=float)

    # remaining: tập chỉ số câu chưa được gán vào chunk nào
    remaining = set(range(n))
    groups: List[List[int]] = []
    while remaining:
        # Chọn anchor có centrality cao nhất (câu "đại diện" nhất trong phần còn lại)
        anchor = max(remaining, key=lambda idx: centrality[idx])
        remaining.remove(anchor)
        # Rank remaining by similarity to anchor
        similars = [
            (idx, float(sim_matrix[anchor, idx]))
            for idx in remaining
            if float(sim_matrix[anchor, idx]) >= anchor_similarity_floor
        ]
        # Sort giảm dần theo similarity đến anchor
        similars.sort(key=lambda x: x[1], reverse=True)
        need = max(0, target_k - 1)
        chosen = [idx for idx, _ in similars[:need]]
        for c in chosen:
            remaining.discard(c)
        # Lưu nhóm: sort theo chỉ số câu để chunk_text giữ nguyên thứ tự xuất hiện trong tài liệu
        groups.append(sorted([anchor] + chosen))

    # Merge trailing undersized group (nhóm cuối quá nhỏ nhập vào nhóm trước)
    if len(groups) >= 2 and len(groups[-1]) < min_sentences_per_chunk:
        groups[-2].extend(groups[-1])
        groups.pop()

    # Forward attach any undersized groups (gom buffer tích luỹ các nhóm nhỏ)
    merged: List[List[int]] = []
    buffer: List[int] = []
    for g in groups:
        if len(g) >= min_sentences_per_chunk:
            if buffer:
                g = sorted(buffer + g)
                buffer = []
            merged.append(g)
        else:
            buffer.extend(g)
    if buffer:
        if merged:
            merged[-1].extend(buffer)
        else:
            merged.append(buffer)

    if len(merged) >= 2 and len(merged[-1]) < min_sentences_per_chunk:
        merged[-2].extend(merged[-1])
        merged.pop()

    final_chunks: List[Tuple[str, str, Optional[str]]] = []
    if collect_metadata:
        # Pre-compute anchor centrality ordering for potential inclusion in metadata
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
            # Build similarity stats within group (pairwise to anchor = first element of g)
            chunk_id_str = f"{doc_id}_anchor{i}"
            meta = {"chunk_id": chunk_id_str, "sent_indices": ",".join(str(x) for x in sorted(set(g))), "n": len(g)}
            if sim_matrix is not None and len(g) > 1:
                anchor = g[0]
                sims_anchor = []
                for idx2 in g[1:]:
                    try:
                        sims_anchor.append(float(sim_matrix[anchor, idx2]))
                    except Exception:
                        continue
                if sims_anchor:
                    import math
                    m = sum(sims_anchor)/len(sims_anchor)
                    mn = min(sims_anchor); mx = max(sims_anchor)
                    var = sum((x-m)**2 for x in sims_anchor)/len(sims_anchor)
                    meta.update({"anchor": anchor, "sim_mean": round(m,4), "sim_min": round(mn,4), "sim_max": round(mx,4), "sim_std": round(math.sqrt(var),4)})
            if centrality_vals:
                try:
                    meta["anchor_centrality"] = round(centrality_vals[g[0]],4)
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
            final_chunks.append((f"{doc_id}_anchor{i}", chunk_text, None))

    del sim_matrix
    gc.collect()

    if not final_chunks:
        return [(f"{doc_id}_fallback", passage_text, None)]

    if not silent:
        log_msg(False, f"Anchor greedy chunks={len(final_chunks)} target={target_k}", 'info', 'group')
    return final_chunks


def semantic_chunk_passage_from_grouping_logic(
    doc_id: str,
    passage_text: str,
    embedding_model: str = "thenlper/gte-base",
    *,
    min_sentences_per_chunk: int = 6,
    anchor_target_sentences: int = 6,
    anchor_similarity_floor: float = 0.0,
    embedding_batch_size: int = 64,
    device: Optional[str] = "cuda",
    silent: bool = False,
    collect_metadata: bool = False,
    **_extra,
) -> List[Tuple[str, str, Optional[str]]]:
    return semantic_grouping_main(
        passage_text=passage_text,
        doc_id=doc_id,
        embedding_model=embedding_model,
        min_sentences_per_chunk=min_sentences_per_chunk,
        anchor_target_sentences=anchor_target_sentences,
        anchor_similarity_floor=anchor_similarity_floor,
        embedding_batch_size=embedding_batch_size,
        device=device,
        silent=silent,
        collect_metadata=collect_metadata,
    )
