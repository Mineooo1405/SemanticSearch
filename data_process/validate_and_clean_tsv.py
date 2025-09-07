import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple, TextIO


def _normalize_text(text: str) -> str:
    """
    Chuẩn hoá một đoạn văn bản ngắn:
      - Ép về str
      - Thay tab bằng khoảng trắng
      - Loại bỏ ký tự xuống dòng
      - Trim hai đầu
      - Nén khoảng trắng thừa (để csv/tsv ổn định)
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return ""
    # Chuẩn hoá các ký tự phân tách trước
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    # Nén khoảng trắng thừa
    s = " ".join(s.split())
    return s.strip()


def _parse_label(raw_label: str):
    """
    Chuyển label sang số, trả về None nếu không hợp lệ.
    Giữ nguyên miền giá trị int (không ép về nhị phân) để tương thích nhiều bài toán ranking.
    """
    try:
        # Cho phép float trong file gốc, nhưng ép về int sau khi hợp lệ
        v = float(str(raw_label).strip())
        if v != v:  # NaN check
            return None
        return int(v)
    except Exception:
        return None


def _read_header(file_obj: TextIO) -> Tuple[str, str, str]:
    """
    Đọc và kiểm tra header. Trả về tuple tên cột (left, right, label) đã xác định.
    Expect: query_text \t chunk_text \t label
    """
    header_line = file_obj.readline()
    if not header_line:
        raise ValueError("File rỗng, không có header.")
    parts = header_line.rstrip("\n").split("\t")
    if len(parts) < 3:
        raise ValueError("Header không hợp lệ, cần tối thiểu 3 cột: query_text\tchunk_text\tlabel")

    # Chuẩn hoá tên cột
    cols = [p.strip().lower() for p in parts]
    # Tìm cột theo tên quen thuộc
    left_candidates = {"query_text", "query", "q", "question", "text_left"}
    right_candidates = {"chunk_text", "passage", "document", "doc", "text", "answer", "text_right"}
    label_candidates = {"label", "relevance", "rel", "y"}

    def _find_col(candidates):
        for idx, name in enumerate(cols):
            if name in candidates:
                return idx, parts[idx]
        return None, None

    li, left_col_name = _find_col(left_candidates)
    ri, right_col_name = _find_col(right_candidates)
    yi, label_col_name = _find_col(label_candidates)

    # Nếu không dò được theo tên, fallback về {0,1,2}
    if li is None or ri is None or yi is None:
        li, ri, yi = 0, 1, 2
        left_col_name, right_col_name, label_col_name = parts[li], parts[ri], parts[yi]

    return left_col_name, right_col_name, label_col_name


def validate_and_clean(
    input_path: Path,
    output_path: Path,
    *,
    report_path: Path,
    drop_unpairable: bool = False,
    chunksize: int = 100_000,
    min_text_len: int = 1,
) -> Dict:
    """
    Kiểm tra và làm sạch TSV đầu ra từ file_mapping.py để tương thích với create_matchzoo_datapacks.py.

    - Xử lý theo chunksize để phù hợp file lớn.
    - Loại các dòng lỗi format, label không hợp lệ, text rỗng sau chuẩn hoá.
    - Tạo thống kê pairability theo query_text (label>0 là pos, <=0 là neg).
    - Tuỳ chọn: drop toàn bộ dòng thuộc các query_text không có cả pos & neg.
    """
    stats = {
        "total_rows": 0,
        "kept_rows": 0,
        "dropped_bad_format": 0,
        "dropped_bad_label": 0,
        "dropped_empty_text": 0,
        "dropped_unpairable": 0,
        "unique_queries": 0,
        "pairable_queries": 0,
        "only_pos_queries": 0,
        "only_neg_queries": 0,
    }

    # Pass 0: đọc header và xác định index cột
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        left_col_name, right_col_name, label_col_name = _read_header(f)

    # Pass 1: quét để đếm pairability theo query_text
    query_label_counts: Dict[str, Dict[str, int]] = {}
    with input_path.open("r", encoding="utf-8", errors="ignore") as src:
        reader = csv.DictReader(src, delimiter="\t")
        for row in reader:
            stats["total_rows"] += 1
            try:
                raw_left = row.get(left_col_name, None)
                raw_right = row.get(right_col_name, None)
                raw_label = row.get(label_col_name, None)
            except Exception:
                stats["dropped_bad_format"] += 1
                continue

            left = _normalize_text(raw_left)
            right = _normalize_text(raw_right)
            label = _parse_label(raw_label)

            if label is None:
                stats["dropped_bad_label"] += 1
                continue
            if not left or not right:
                stats["dropped_empty_text"] += 1
                continue
            if len(left) < min_text_len or len(right) < min_text_len:
                stats["dropped_empty_text"] += 1
                continue

            # Tính pairability trên dữ liệu đủ điều kiện cơ bản
            cnt = query_label_counts.setdefault(left, {"pos": 0, "neg": 0})
            if label > 0:
                cnt["pos"] += 1
            else:
                cnt["neg"] += 1

    stats["unique_queries"] = len(query_label_counts)

    # Phân loại query theo pairability
    unpairable_queries = set()
    for q, c in query_label_counts.items():
        if c["pos"] > 0 and c["neg"] > 0:
            stats["pairable_queries"] += 1
        elif c["pos"] > 0 and c["neg"] == 0:
            stats["only_pos_queries"] += 1
            unpairable_queries.add(q)
        elif c["neg"] > 0 and c["pos"] == 0:
            stats["only_neg_queries"] += 1
            unpairable_queries.add(q)
        else:
            # Cả pos và neg đều 0 không thể xảy ra nếu đã đi qua các kiểm tra trên
            unpairable_queries.add(q)

    # Pass 2: ghi file sạch
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", errors="ignore") as src, \
            output_path.open("w", encoding="utf-8", errors="ignore", newline="") as out:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["query_text", "chunk_text", "label"])  # Header chuẩn hoá cho create_matchzoo_datapacks

        for row in reader:
            try:
                raw_left = row.get(left_col_name, None)
                raw_right = row.get(right_col_name, None)
                raw_label = row.get(label_col_name, None)
            except Exception:
                # Đếm bad_format lại ở pass 2 là dư thừa; bỏ qua để tránh double-count
                continue

            left = _normalize_text(raw_left)
            right = _normalize_text(raw_right)
            label = _parse_label(raw_label)

            if label is None:
                continue
            if not left or not right:
                continue
            if len(left) < min_text_len or len(right) < min_text_len:
                continue

            if drop_unpairable and left in unpairable_queries:
                stats["dropped_unpairable"] += 1
                continue

            writer.writerow([left, right, label])
            stats["kept_rows"] += 1

    # Ghi báo cáo JSON + TSV pairability
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "drop_unpairable": bool(drop_unpairable),
        "stats": stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pairability_tsv = output_path.with_suffix("")
    pairability_tsv = pairability_tsv.with_name(output_path.stem + ".pairability.tsv")
    with pairability_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_text", "pos", "neg", "total", "status"])
        for q, c in query_label_counts.items():
            status = (
                "pairable" if (c["pos"] > 0 and c["neg"] > 0)
                else ("only_pos" if c["pos"] > 0 else ("only_neg" if c["neg"] > 0 else "empty"))
            )
            w.writerow([q, c["pos"], c["neg"], c["pos"] + c["neg"], status])

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Kiểm tra và làm sạch TSV từ file_mapping.py; đảm bảo hợp lệ cho create_matchzoo_datapacks.py"
        )
    )
    p.add_argument("input_tsv", type=Path, help="File TSV đầu vào (query_text, chunk_text, label)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="File TSV đầu ra sạch (mặc định: <input>.clean.tsv)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="File JSON lưu thống kê (mặc định: <output>.report.json)",
    )
    p.add_argument(
        "--drop-unpairable",
        action="store_true",
        help="Loại bỏ toàn bộ dòng thuộc query không có cả pos và neg",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunksize đọc file (hiện tại dùng streaming, tham số để tương thích về sau)",
    )
    p.add_argument(
        "--min-text-len",
        type=int,
        default=1,
        help="Độ dài tối thiểu sau chuẩn hoá cho query_text và chunk_text",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_path = args.input_tsv
    if not input_path.exists():
        print(f"Lỗi: Không tìm thấy file: {input_path}")
        return 2

    output_path = args.output or input_path.with_suffix("")
    if output_path == input_path.with_suffix(""):
        output_path = output_path.with_name(input_path.name + ".clean.tsv")

    report_path = args.report or output_path.with_suffix("")
    if report_path == output_path.with_suffix(""):
        report_path = report_path.with_name(output_path.name + ".report.json")

    try:
        report = validate_and_clean(
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            drop_unpairable=bool(args.drop_unpairable),
            chunksize=int(args.chunksize),
            min_text_len=int(args.min_text_len),
        )
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return 1

    # Tóm tắt ngắn
    s = report["stats"]
    print("✓ Hoàn tất làm sạch TSV")
    print(f"  Input : {report['input']}")
    print(f"  Output: {report['output']}")
    print(f"  Kept  : {s['kept_rows']:,} / {s['total_rows']:,}")
    print(
        "  Drops : bad_format=%d, bad_label=%d, empty_text=%d, unpairable=%d"
        % (s["dropped_bad_format"], s["dropped_bad_label"], s["dropped_empty_text"], s["dropped_unpairable"]) 
    )
    print(
        "  Query : unique=%d, pairable=%d, only_pos=%d, only_neg=%d"
        % (s["unique_queries"], s["pairable_queries"], s["only_pos_queries"], s["only_neg_queries"]) 
    )
    print(f"  Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


