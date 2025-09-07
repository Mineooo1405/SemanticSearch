from typing import List, Tuple, Optional

"""
TextTiling-LLM - Thuật toán chia văn bản kiểu "nhìn trái nhìn phải"

Đây là thuật toán baseline được mô tả trong bài báo COLING 2025. Ý tưởng đơn giản:

1) BƯỚC ĐẦU: Chia văn bản thành từng câu riêng biệt

2) XÉT TỪNG VỊ TRÍ CẮT TIỀM NĂNG:
   - Đứng ở giữa hai câu bất kỳ, ví dụ giữa câu 15 và câu 16
   - "Nhìn về bên trái": lấy 5 câu trước đó (câu 11-15) 
   - "Nhìn về bên phải": lấy 5 câu sau đó (câu 16-20)

3) SO SÁNH HAI BÊN:
   - Biến tất cả câu thành vector số (embedding)
   - Tính độ giống nhau giữa MỌI CẶP câu trái-phải (25 cặp nếu mỗi bên 5 câu)
   - Lấy độ giống nhau CAO NHẤT trong 25 cặp đó

4) QUYẾT ĐỊNH CẮT:
   - Nếu độ giống nhau cao → hai bên nói về cùng chủ đề → KHÔNG cắt
   - Nếu độ giống nhau thấp → hai bên nói về chủ đề khác → CẮT ở đây
   - Điểm cắt = 1 - (độ giống nhau cao nhất)
   - Điểm càng cao → càng nên cắt

5) LỌC KẾT QUẢ:
   - Chỉ cắt ở những vị trí có điểm ≥ ngưỡng (ví dụ 0.5)
   - Đảm bảo hai điểm cắt không quá gần nhau (tối thiểu cách 10 câu)
   - Ưu tiên điểm cắt có điểm số cao hơn

KHÁC BIỆT VỚI TEXTTILING TRUYỀN THỐNG:
- TextTiling cũ: đếm từ chung giữa hai bên
- TextTiling-LLM: dùng AI để hiểu nghĩa câu, so sánh nghĩa thay vì đếm từ
"""

import numpy as np
from Tool.Sentence_Segmenter import extract_sentences_spacy
from Method.semantic_common import normalize_device, embed_sentences_batched


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    if arr is None or arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return arr / norms


def _pairwise_max_cos(left: np.ndarray, right: np.ndarray) -> float:
    """
    Tìm độ giống nhau cao nhất giữa hai nhóm câu.
    
    Ví dụ: nhóm trái có 5 câu, nhóm phải có 5 câu
    → Tính độ giống 5x5=25 cặp, lấy giá trị cao nhất
    
    Nếu có cặp nào rất giống nhau → hai bên có liên quan → không nên cắt
    """
    if left.size == 0 or right.size == 0:
        return 0.0
    S = left @ right.T  # Ma trận 5x5 chứa độ giống của 25 cặp
    return float(S.max()) if S.size > 0 else 0.0


def texttiling_llm_splitter(
    doc_id: str,
    passage_text: str,
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    k_window: int = 5,
    threshold: float = 0.5,
    min_boundary_spacing: int = 10,
    min_first_boundary_index: int = 5,
    # controller compatibility
    silent: bool = False,
    output_dir: Optional[str] = None,
    collect_metadata: bool = False,
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Hàm chính - chia văn bản bằng thuật toán TextTiling-LLM.
    
    QUY TRÌNH:
    1) Chia văn bản thành từng câu
    2) Biến tất cả câu thành vector AI (embedding)
    3) Với mỗi vị trí giữa 2 câu: lấy k câu trước và k câu sau
    4) Tính độ giống nhau cao nhất giữa hai nhóm k câu  
    5) Điểm cắt = 1 - độ giống nhau cao nhất
    6) Cắt ở những vị trí có điểm ≥ ngưỡng, không quá gần nhau
    """
    sentences = extract_sentences_spacy(passage_text)
    if not sentences:
        return []
    if len(sentences) <= 1:
        return [(f"{doc_id}_chunk0", passage_text, None)]
    n = len(sentences)
    dev = normalize_device(device)
    embs = embed_sentences_batched(sentences, embedding_model, base_batch_size=32, device=dev, silent=True)
    if embs is None or (hasattr(embs, 'size') and embs.size == 0):
        return [(f"{doc_id}_chunk0", passage_text, None)]
    if isinstance(embs, list):
        embs = np.array(embs, dtype=float)
    embs = _l2_normalize_rows(embs)
    k = max(1, int(k_window))
    scores = np.zeros((n - 1,), dtype=float)  # Điểm cắt cho từng vị trí giữa 2 câu
    
    # Xét từng vị trí cắt tiềm năng giữa câu i và i+1
    for i in range(n - 1):
        # Nhóm trái: k câu trước vị trí cắt (kết thúc ở câu i)
        a0 = max(0, i - k + 1)  # Câu bắt đầu nhóm trái
        a1 = i + 1              # Câu kết thúc nhóm trái (không bao gồm)
        
        # Nhóm phải: k câu sau vị trí cắt (bắt đầu từ câu i+1)
        b0 = i + 1              # Câu bắt đầu nhóm phải  
        b1 = min(n, i + 1 + k)  # Câu kết thúc nhóm phải (không bao gồm)
        
        left = embs[a0:a1]   # Vector của nhóm trái
        right = embs[b0:b1]  # Vector của nhóm phải
        
        max_cos = _pairwise_max_cos(left, right)  # Độ giống cao nhất giữa 2 nhóm
        scores[i] = 1.0 - max_cos  # Điểm cắt: càng khác nhau càng cao
    thr = float(threshold)
    cand = [i + 1 for i, v in enumerate(scores.tolist()) if v >= thr]  # Vị trí có điểm ≥ ngưỡng
    
    # Sắp xếp theo điểm số từ cao xuống thấp (ưu tiên điểm cắt tốt nhất)
    cand_sorted = sorted(cand, key=lambda b: scores[b - 1], reverse=True)
    selected: List[int] = []
    
    # Chọn điểm cắt theo nguyên tắc: không quá gần nhau, không quá sớm
    for b in cand_sorted:
        # Bỏ qua nếu cắt quá sớm (cần ít nhất vài câu đầu)
        if b < int(min_first_boundary_index):
            continue
        # Bỏ qua nếu quá gần điểm cắt đã chọn (tránh cắt liên tục)    
        if all(abs(b - x) >= int(min_boundary_spacing) for x in selected):
            selected.append(b)
    
    boundaries = sorted(set(selected))  # Các vị trí cắt cuối cùng
    all_bounds = boundaries + [n]
    chunks: List[Tuple[str, str, Optional[str]]] = []
    cursor = 0
    for idx, b in enumerate(all_bounds):
        if b <= cursor:
            continue
        ctext = " ".join(sentences[cursor:b])
        cid = f"{doc_id}_chunk{idx}"
        if collect_metadata:
            import json
            meta = {
                "chunk_id": cid,
                "sent_indices": ",".join(str(x) for x in range(cursor, b)),
                "n": b - cursor,
                "k_window": k,
                "threshold": thr,
            }
            chunks.append((cid, ctext, json.dumps(meta, ensure_ascii=False)))
        else:
            chunks.append((cid, ctext, None))
        cursor = b
    return chunks


