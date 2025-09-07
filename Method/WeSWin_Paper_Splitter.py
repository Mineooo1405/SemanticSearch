from typing import List, Tuple, Optional

"""
WeSWin (Weighted Sliding Windows) - Thuật toán chia văn bản thông minh

Đây là thuật toán được mô tả trong bài báo COLING 2025. Ý tưởng chính:

1) BƯỚC ĐẦU: Cắt văn bản thành từng câu riêng biệt

2) TẠO CÁC CỬA SỔ TRƯỢT:
   - Tưởng tượng bạn có một "khung nhìn" để quan sát 30 câu cùng lúc
   - Khung này trượt qua văn bản: lần đầu nhìn câu 1-30, lần sau nhìn câu 7-36, tiếp theo 13-42...
   - Mỗi lần trượt, khung di chuyển 6 câu (gọi là "stride")

3) DỰ ĐOÁN ĐIỂM CẮT:
   - Trong mỗi khung 30 câu, thuật toán tính xem "sau câu nào nên cắt đoạn?"
   - Phương pháp: so sánh độ giống nhau giữa câu liền kề - nếu hai câu rất khác nhau → có thể cắt ở đây
   - Kết quả: mỗi câu có một "điểm số cắt" từ 0 đến 1

4) GỘP KẾT QUẢ THÔNG MINH:
   - Vì mỗi câu xuất hiện trong nhiều khung khác nhau, ta có nhiều "điểm số cắt" cho cùng 1 câu
   - Câu ở giữa khung được tin tưởng hơn (có thông tin đầy đủ 2 bên)
   - Câu ở rìa khung ít đáng tin (thiếu thông tin 1 bên)
   - → Gộp điểm số có trọng số: câu giữa khung có trọng số cao, câu rìa có trọng số thấp

5) QUYẾT ĐỊNH CUỐI CÙNG:
   - Sau khi có điểm số cuối cùng cho mỗi câu, cắt tại những câu có điểm ≥ ngưỡng (ví dụ 0.5)

KHÁC BIỆT VỚI BÀI BÁO GỐC:
- Bài báo gốc dùng mô hình AI được huấn luyện sẵn để tính điểm số cắt
- Ở đây ta dùng phương pháp đơn giản: tính độ giống nhau giữa câu kề nhau
- Khi nào có mô hình AI sẵn, chỉ cần thay thế phần tính điểm này là xong
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


def _window_starts(n_sent: int, stride_k: int) -> List[int]:
    """
    Tính vị trí bắt đầu của các cửa sổ trượt.
    
    Ví dụ: có 100 câu, stride=6
    → Cửa sổ 1: bắt đầu câu 0
    → Cửa sổ 2: bắt đầu câu 6  
    → Cửa sổ 3: bắt đầu câu 12
    → ...
    """
    starts = []
    i = 0
    while i < n_sent:
        starts.append(i)
        if i == n_sent - 1:
            break
        i += max(1, int(stride_k))
    return starts


def _pack_limit(sentences: List[str], start: int, *, max_sentences: int) -> int:
    """
    Giới hạn độ dài cửa sổ.
    
    Bài báo gốc giới hạn theo số token, ở đây ta giới hạn theo số câu cho đơn giản.
    Ví dụ: bắt đầu từ câu 10, tối đa 30 câu → kết thúc ở câu 40 (hoặc hết văn bản)
    """
    return min(len(sentences), start + max(1, int(max_sentences)))


def _linear_weight(i: int, m: int, k: int, eps: float) -> float:
    """
    Tính trọng số theo vị trí trong cửa sổ (kiểu tuyến tính).
    
    Câu ở giữa cửa sổ → trọng số cao (tin cậy)
    Câu ở rìa cửa sổ → trọng số thấp (ít tin cậy)
    
    Ví dụ: cửa sổ 10 câu, câu số 5 (giữa) có trọng số cao nhất
    """
    di = min(i - 1, m - i)  # Khoảng cách tới rìa gần nhất
    return float(eps + (1.0 - eps) * (min(di, k) / float(max(1, k))))


def _poly_weight(i: int, m: int, k: int, p: int, eps: float) -> float:
    """
    Tính trọng số theo vị trí (kiểu đa thức).
    
    Giống linear nhưng "cong" hơn - có thể tăng/giảm nhanh hoặc chậm tùy tham số p.
    p=1: giống linear, p>1: tăng chậm rồi nhanh, p<1: tăng nhanh rồi chậm
    """
    di = min(i - 1, m - i)
    x = min(di, k) / float(max(1, k))
    return float(eps + (1.0 - eps) * (1.0 - (1.0 - x) ** max(1, int(p))))


def _aggregate(per_window_probs: List[Tuple[int, int, List[float]]], *,
               weighting: str = "linear",
               k: int = 8,
               eps: float = 0.1,
               poly_p: int = 2) -> List[float]:
    """
    Gộp điểm số cắt từ nhiều cửa sổ cho từng câu.
    
    Mỗi câu có thể xuất hiện trong nhiều cửa sổ → có nhiều điểm số cắt khác nhau.
    Ta gộp chúng bằng trọng số: điểm từ cửa sổ nào câu đó ở giữa sẽ có trọng số cao hơn.
    """
    if not per_window_probs:
        return []
    n_global = max(e for (_, e, _p) in per_window_probs)
    agg = np.zeros((n_global,), dtype=float)
    wsum = np.zeros((n_global,), dtype=float)
    for start, end, p in per_window_probs:
        m = end - start
        if m <= 0:
            continue
        w = []
        for idx in range(m):
            pos = idx + 1
            if weighting == "linear":
                wv = _linear_weight(pos, m, k, eps)
            elif weighting == "poly":
                wv = _poly_weight(pos, m, k, poly_p, eps)
            else:
                wv = 1.0
            w.append(float(wv))
        w = np.array(w, dtype=float)
        p_arr = np.array(p, dtype=float)
        L = min(m, p_arr.size)
        if L <= 0:
            continue
        agg[start:start + L] += p_arr[:L] * w[:L]
        wsum[start:start + L] += w[:L]
    wsum[wsum == 0] = 1e-9
    out = (agg / wsum).tolist()
    if len(out) > 0:
        out[-1] = 0.0
    return out


def _unsup_window_probs(sentences: List[str], start: int, end: int,
                        *, embedding_model: str, device: Optional[str]) -> List[float]:
    """
    Tính điểm số cắt cho từng câu trong một cửa sổ (phiên bản đơn giản).
    
    Bài báo gốc dùng mô hình AI được huấn luyện. Ở đây ta dùng cách đơn giản:
    1) Biến mỗi câu thành vector số (embedding)
    2) Tính độ giống nhau giữa câu liền kề  
    3) Nếu hai câu rất khác nhau → có thể cắt ở đây
    4) Chuyển độ khác nhau thành điểm số từ 0-1
    
    VẤN ĐỀ VỚI SIGMOID(-Z): 
    - Sigmoid(-z_score) cho kết quả xoay quanh 0.5
    - Threshold 0.5 sẽ cắt ~50% vị trí → quá nhiều cuts!
    - Cần dùng percentile-based approach để calibrated hơn
    """
    seg = sentences[start:end]
    if len(seg) <= 1:
        return [0.0] * len(seg)
    dev = normalize_device(device)
    embs = embed_sentences_batched(seg, embedding_model, base_batch_size=32, device=dev, silent=True)
    if embs is None or (hasattr(embs, 'size') and embs.size == 0):
        return [0.0] * len(seg)
    if isinstance(embs, list):
        embs = np.array(embs, dtype=float)
    embs = _l2_normalize_rows(embs)
    sims = [float(embs[i] @ embs[i + 1]) for i in range(len(seg) - 1)]
    sims_arr = np.array(sims, dtype=float)
    
    # CÁCH CŨNG (sigmoid-based) - quá nhạy cảm
    # mu = float(sims_arr.mean())
    # sd = float(sims_arr.std() + 1e-9)
    # z = (sims_arr - mu) / sd
    # from math import exp
    # probs = [1.0 / (1.0 + exp(zv)) for zv in z]
    
    # CÁCH MỚI: Dùng percentile để chỉ cắt ở những vị trí thực sự khác biệt nhất
    # Chỉ top 20% vị trí có similarity thấp nhất mới được coi là có khả năng cắt cao
    percentile_80 = float(np.percentile(sims_arr, 80))  # 80% các similarity ≥ giá trị này
    
    # Mapping similarity → probability: 
    # - sim ≥ p80: prob = 0.0-0.3 (ít khả năng cắt)  
    # - sim < p80: prob = 0.3-1.0 (khả năng cắt tăng dần)
    probs = []
    for sim in sims:
        if sim >= percentile_80:
            # Sim cao → ít cắt (0.0-0.3)
            normalized = (sim - percentile_80) / (1.0 - percentile_80 + 1e-9)
            prob = 0.3 * (1.0 - normalized)  # sim càng cao → prob càng thấp
        else:
            # Sim thấp → nhiều cắt (0.3-1.0)  
            normalized = sim / (percentile_80 + 1e-9)
            prob = 0.3 + 0.7 * (1.0 - normalized)  # sim càng thấp → prob càng cao
        probs.append(float(prob))
    
    probs.append(0.0)  # Câu cuối không bao giờ cắt
    return probs


def weswin_paper_splitter(
    doc_id: str,
    passage_text: str,
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    # sliding windows
    stride_k: int = 6,
    max_sentences_per_window: int = 30,
    # weighting
    weighting: str = "linear",  # linear | poly | uniform
    weight_eps: float = 0.1,
    weight_k: int = 8,
    weight_poly_p: int = 2,
    # decision
    decision_threshold: float = 0.7,
    smooth_window: int = 5,
    min_boundary_spacing: int = 10,
    min_first_boundary_index: int = 8,
    # controller compatibility
    silent: bool = False,
    output_dir: Optional[str] = None,
    collect_metadata: bool = False,
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Hàm chính - chia văn bản bằng thuật toán WeSWin.
    
    QUY TRÌNH:
    1) Chia văn bản thành từng câu
    2) Tạo các cửa sổ trượt (mỗi cửa sổ chứa 30 câu, trượt 6 câu mỗi lần)  
    3) Với mỗi cửa sổ: tính điểm số cắt cho từng câu trong cửa sổ đó
    4) Gộp tất cả điểm số (có trọng số) để có điểm cuối cùng cho mỗi câu
    5) Cắt ở những câu có điểm ≥ ngưỡng (mặc định 0.5)
    """
    sentences = extract_sentences_spacy(passage_text)
    if not sentences:
        return []
    if len(sentences) <= 1:
        return [(f"{doc_id}_chunk0", passage_text, None)]
    n = len(sentences)
    starts = _window_starts(n, stride_k)
    per_window_probs: List[Tuple[int, int, List[float]]] = []
    for st in starts:
        ed = _pack_limit(sentences, st, max_sentences=max_sentences_per_window)
        if ed - st <= 0:
            continue
        probs = _unsup_window_probs(sentences, st, ed, embedding_model=embedding_model, device=device)
        if len(probs) < (ed - st):
            probs = probs + [0.0] * ((ed - st) - len(probs))
        elif len(probs) > (ed - st):
            probs = probs[:(ed - st)]
        per_window_probs.append((st, ed, probs))

    # Gộp các dự đoán từ nhiều cửa sổ → xác suất cuối cùng cho mỗi câu
    final_probs = _aggregate(
        per_window_probs,
        weighting=weighting,
        k=weight_k,
        eps=weight_eps,
        poly_p=weight_poly_p,
    )

    # === HẬU XỬ LÝ: làm mượt + local-max + NMS khoảng cách tối thiểu ===
    # 1) Làm mượt xác suất để giảm nhiễu (moving average)
    try:
        W = max(1, int(smooth_window))
    except Exception:
        W = 5
    if W > 1 and len(final_probs) > 0:
        kernel = np.ones((W,), dtype=float) / float(W)
        smoothed = np.convolve(np.array(final_probs, dtype=float), kernel, mode='same').tolist()
    else:
        smoothed = list(final_probs)

    # 2) Chọn các đỉnh cục bộ vượt ngưỡng (local maxima)
    candidates: List[Tuple[int, float]] = []  # (boundary_after_index, score)
    thr = float(decision_threshold)
    for i_edge in range(1, n):
        score = float(smoothed[i_edge - 1])
        if score < thr:
            continue
        left = smoothed[i_edge - 2] if i_edge - 2 >= 0 else -1e9
        right = smoothed[i_edge] if i_edge < len(smoothed) else -1e9
        # local max hoặc plateau giữa hai giá trị kề
        if score >= left and score >= right:
            candidates.append((i_edge, score))

    # 3) Non‑Maximum Suppression theo khoảng cách câu tối thiểu
    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected: List[int] = []
    min_space = max(1, int(min_boundary_spacing))
    min_first = max(1, int(min_first_boundary_index))
    for idx, sc in candidates_sorted:
        if idx < min_first:
            continue
        if all(abs(idx - s) >= min_space for s in selected):
            selected.append(idx)
    boundaries = sorted(selected)
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
                "weswin_weighting": weighting,
                "weswin_stride": stride_k,
            }
            chunks.append((cid, ctext, json.dumps(meta, ensure_ascii=False)))
        else:
            chunks.append((cid, ctext, None))
        cursor = b
    return chunks


