"""
Naive character-length based splitter.

Yêu cầu người dùng: "doc có 1000 chữ (ký tự) và chunk_size = 600 => 2 chunk: 600 ký tự đầu, phần còn lại".

Đây là phiên bản đơn giản nhất:
- Chỉ cắt theo số ký tự cố định (chunk_size).
- Không semantics, không đếm token, không tách câu, không overlap (trừ khi bạn bật tuỳ chọn).
- Thời gian chạy O(L) với L = độ dài chuỗi.

Hàm trả về list tuple: (chunk_id, chunk_text, metadata_json_or_None)
Metadata (nếu collect_metadata=True): start_char, end_char (exclusive), length.

Tuỳ chọn overlap (số ký tự lặp lại) được hỗ trợ nhẹ nếu cần mở rộng (mặc định 0).
"""
from __future__ import annotations

import json
from typing import List, Tuple, Optional

Chunk = Tuple[str, str, Optional[str]]

def chunk_passage_char_splitter(
    doc_id: str,
    passage_text: str,
    chunk_size: int = 600,
    overlap: int = 0,
    collect_metadata: bool = False,
    silent: bool = False,
) -> List[Chunk]:
    if not passage_text:
        return []

    if chunk_size <= 0:
        # Trường hợp xấu: trả về nguyên văn như một chunk
        return [(f"{doc_id}_chunk0", passage_text, json.dumps({
            "chunk_id": f"{doc_id}_chunk0", "start_char": 0, "end_char": len(passage_text), "length": len(passage_text)
        }, ensure_ascii=False) if collect_metadata else None)]

    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    step = chunk_size - overlap if chunk_size - overlap > 0 else chunk_size

    text_len = len(passage_text)
    chunks: List[Chunk] = []
    idx = 0
    chunk_index = 0

    if not silent:
        print(f"[char_split] doc={doc_id} len={text_len} chunk_size={chunk_size} overlap={overlap}")

    while idx < text_len:
        end = idx + chunk_size
        piece = passage_text[idx:end]
        chunk_id = f"{doc_id}_chunk{chunk_index}"
        meta: Optional[str] = None
        if collect_metadata:
            meta_obj = {
                "chunk_id": chunk_id,
                "start_char": idx,
                "end_char": min(end, text_len),
                "length": len(piece),
            }
            meta = json.dumps(meta_obj, ensure_ascii=False)
        chunks.append((chunk_id, piece, meta))
        if end >= text_len:
            break
        idx += step
        chunk_index += 1

    return chunks

# Alias đồng bộ hoá với naming chung (nếu controller đòi hàm chunk_passage_text_splitter)

def chunk_passage_text_splitter(
    doc_id: str,
    passage_text: str,
    chunk_size: int = 600,
    overlap: int = 0,
    collect_metadata: bool = False,
    silent: bool = False,
    **_ignored,
) -> List[Chunk]:
    return chunk_passage_char_splitter(
        doc_id=doc_id,
        passage_text=passage_text,
        chunk_size=chunk_size,
        overlap=overlap,
        collect_metadata=collect_metadata,
        silent=silent,
    )

if __name__ == "__main__":
    demo = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor. Cras elementum ultrices diam. Maecenas ligula massa, varius a, semper congue, euismod non, mi."""
    res = chunk_passage_char_splitter("demo", demo, chunk_size=60, collect_metadata=True)
    for r in res:
        print(r[0], len(r[1]), r[2])
