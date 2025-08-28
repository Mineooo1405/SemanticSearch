from typing import List, Tuple, Optional


"""
Coling2025_RobustDP_Splitter

Wrapper sử dụng thuật toán robust DP segmentation (unified boundary signal + DP)
đã được triển khai trong `Method/Semantic_Splitter_Optimized.py`.

Tài liệu tham khảo: doc/2025.coling-industry.67.pdf

API xuất ra tương thích với controller: chunk_passage_text_splitter_coling2025(...)
"""

from .Semantic_Splitter_Optimized import semantic_splitter_main


def chunk_passage_text_splitter_coling2025(
    doc_id: str,
    passage_text: str,
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    # ràng buộc biên cơ bản
    min_boundary_spacing: int = 8,
    min_first_boundary_index: int = 6,
    # tham số unified signal + DP (có thể override qua kwargs)
    r_window: int = 4,
    r_w_drop: float = 0.6,
    r_w_drift: float = 0.3,
    r_w_valley: float = 0.1,
    r_tau: float = 1.0,
    rdp_penalty: float = 0.65,
    # hậu xử lý tinh chỉnh biên
    refine_max_shift: int = 2,
    refine_alpha: float = 1.0,
    refine_beta: float = 0.5,
    refine_balance: float = 0.05,
    # điều khiển độ dài không cứng (tùy chọn)
    min_chunk_len: Optional[int] = 8,
    max_chunk_len: Optional[int] = 24,
    silent: bool = False,
    collect_metadata: bool = False,
    **legacy_kwargs,
) -> List[Tuple[str, str, Optional[str]]]:
    """Chunk văn bản với robust DP segmentation.

    Cho phép override các tham số qua legacy_kwargs nếu cần tinh chỉnh.
    """
    params = dict(legacy_kwargs)
    # Bắt buộc sử dụng robust_dp
    params["algorithm"] = "robust_dp"
    # Unified signal weights
    params.setdefault("r_window", int(r_window))
    params.setdefault("r_w_drop", float(r_w_drop))
    params.setdefault("r_w_drift", float(r_w_drift))
    params.setdefault("r_w_valley", float(r_w_valley))
    params.setdefault("r_tau", float(r_tau))
    params.setdefault("rdp_penalty", float(rdp_penalty))
    # Refinement
    params.setdefault("refine_max_shift", int(refine_max_shift))
    params.setdefault("refine_alpha", float(refine_alpha))
    params.setdefault("refine_beta", float(refine_beta))
    params.setdefault("refine_balance", float(refine_balance))
    # Length post-process
    params.setdefault("min_chunk_len", min_chunk_len)
    params.setdefault("max_chunk_len", max_chunk_len)

    return semantic_splitter_main(
        doc_id=doc_id,
        passage_text=passage_text,
        embedding_model=embedding_model,
        device=device,
        min_boundary_spacing=min_boundary_spacing,
        min_first_boundary_index=min_first_boundary_index,
        silent=silent,
        collect_metadata=collect_metadata,
        **params,
    )


if __name__ == "__main__":
    # Minimal CLI: đọc text từ stdin và in số chunk + độ dài
    import sys
    txt = sys.stdin.read()
    res = chunk_passage_text_splitter_coling2025("stdin", txt, silent=False)
    print(f"chunks={len(res)}")
    for i, (_cid, ctext, _meta) in enumerate(res):
        print(f"[{i}] len_chars={len(ctext)} len_words={len(ctext.split())}")


