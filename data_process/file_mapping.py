import os
import re
import argparse
from pathlib import Path
from typing import Dict

def parse_topics(topics_file_path: str) -> Dict[str, str]:
    """Đọc topics theo kiểu streaming để tiết kiệm bộ nhớ.

    Trả về dict: {query_id: query_text}, ưu tiên desc + ". " + narr; nếu thiếu desc thì dùng title.
    """
    def _clean(text: str) -> str:
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
        return re.sub(r"\s+", " ", text)

    re_num = re.compile(r"<num>\s*Number:\s*(\d+)")
    re_desc = re.compile(r"<desc>\s*Description:(.*?)(?=<narr>|</top>|\Z)", re.DOTALL)
    re_narr = re.compile(r"<narr>\s*Narrative:(.*?)(?=</top>|\Z)", re.DOTALL)
    re_title = re.compile(r"<title>\s*(.*?)(?=<desc>|<narr>|</top>|\Z)", re.DOTALL)

    topics: Dict[str, str] = {}
    buf = []
    inside = False

    with open(topics_file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "<top>" in line:
                inside = True
                buf = [line]
                continue
            if inside:
                buf.append(line)
                if "</top>" in line:
                    block = "".join(buf)
                    try:
                        num_m = re_num.search(block)
                        if not num_m:
                            inside = False
                            buf = []
                            continue
                        qid = num_m.group(1).strip()
                        desc_m = re_desc.search(block)
                        narr_m = re_narr.search(block)
                        title_m = re_title.search(block)
                        desc = _clean(desc_m.group(1)) if desc_m else ""
                        narr = _clean(narr_m.group(1)) if narr_m else ""
                        title = _clean(title_m.group(1)) if title_m else ""

                        if desc:
                            qt = f"{desc}. {narr}".strip()
                        elif title:
                            qt = f"{title}. {narr}".strip()
                        else:
                            qt = narr or ""
                        qt = _clean(qt)
                        if qt:
                            topics[qid] = qt
                    finally:
                        inside = False
                        buf = []

    return topics

def add_query_text_to_tsv(
    file_final_path: str,
    file_topics_path: str,
    output_path: str,
    *,
    quiet: bool = True,
    max_warnings: int = 20,
):
    """
    Thêm query_text vào file TSV từ topics file
    
    Args:
        file_final_path: Đường dẫn đến file TSV đã filter (query_id, chunk_text, label)
        file_topics_path: Đường dẫn đến file topics chứa query descriptions
        output_path: Đường dẫn file output
    """
    # Bước 1: Lấy dict mapping {query_id: query_text}
    topics = parse_topics(file_topics_path)
    if not topics:
        print("Không tìm thấy query_text từ topics. Kiểm tra lại file topics.")
        return False

    if not quiet:
        print(f"Đã load {len(topics)} queries từ topics file")

    # Bước 2: Đọc file final, ghi file mới
    processed_lines = 0
    skipped_lines = 0
    warn_count = 0

    # Tạo thư mục output nếu cần
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_final_path, 'r', encoding='utf-8', errors='ignore') as src, \
         open(output_path, 'w', encoding='utf-8', errors='ignore') as out:

        # Bỏ qua header (nếu có)
        header = src.readline()
        out.write("query_text\tchunk_text\tlabel\n")  # Header mới

        line_num = 1
        for line in src:
            line_num += 1
            line = line.rstrip('\n')
            if not line:
                continue

            # Xử lý trường hợp chunk_text có chứa tab: giữ cột đầu là query_id, cột cuối là label, phần giữa gộp lại
            parts = line.split('\t')
            if len(parts) < 3:
                if warn_count < max_warnings and not quiet:
                    print(f"Dòng lỗi format ở dòng {line_num}: {line[:200]}...")
                warn_count += 1
                skipped_lines += 1
                continue

            query_id = parts[0].strip()
            label = parts[-1].strip()
            chunk_text = '\t'.join(parts[1:-1])
            # Loại bỏ tab trong chunk_text để đảm bảo TSV 3 cột chuẩn
            if '\t' in chunk_text:
                chunk_text = chunk_text.replace('\t', ' ')
            # Nén khoảng trắng thừa
            chunk_text = re.sub(r"\s+", " ", chunk_text).strip()

            query_text = topics.get(query_id, "")
            if not query_text:
                if warn_count < max_warnings and not quiet:
                    print(f"Không tìm thấy query_text với query_id={query_id} ở dòng {line_num}")
                warn_count += 1
                skipped_lines += 1
                continue

            out.write(f"{query_text}\t{chunk_text}\t{label}\n")
            processed_lines += 1

            # In tiến độ nhẹ để tránh flood console
            if not quiet and processed_lines % 200000 == 0:
                print(f"Đã xử lý {processed_lines:,} dòng...")

    if warn_count > max_warnings and not quiet:
        print(f"(Đã ẩn {warn_count - max_warnings} cảnh báo tương tự)")

    print(f"Đã xử lý: {processed_lines} dòng")
    print(f"Bỏ qua: {skipped_lines} dòng")
    print(f"Đã tạo file: {output_path}")
    return True


def main():
    file_final_path = input("Nhập đường dẫn đến file đã filter: ").strip()
    file_topics_path = input("Nhập đường dẫn đến file topics: ").strip()
    output_path = input("Đường dẫn file output [final_with_querytext.tsv]: ").strip() or "final_with_querytext.tsv"

    try:
        add_query_text_to_tsv(file_final_path, file_topics_path, output_path, quiet=False)
    except Exception as e:
        print(f"Có lỗi xảy ra khi mapping: {e}")


def _cli():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Thêm query_text vào file TSV từ topics file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python file_mapping.py input.tsv topics.robust04.txt -o output.tsv
  python file_mapping.py filtered_chunks.tsv ../robustirdata/topics.robust04.txt --output final_result.tsv
        """
    )
    
    parser.add_argument("input_tsv", type=Path,
                       help="File TSV đầu vào (query_id, chunk_text, label)")
    
    parser.add_argument("topics_file", type=Path, 
                       help="File topics chứa query descriptions")
    
    parser.add_argument("-o", "--output", type=Path,
                       default="final_with_querytext.tsv",
                       help="File TSV đầu ra (default: final_with_querytext.tsv)")
    parser.add_argument("--quiet", action="store_true", help="Giảm log để tránh treo IDE (khuyến nghị)")
    parser.add_argument("--max-warnings", type=int, default=20, help="Số cảnh báo tối đa sẽ in ra (mặc định 20)")
    
    args = parser.parse_args()
    
    # Kiểm tra file input
    if not args.input_tsv.exists():
        print(f"Error: Input file không tồn tại: {args.input_tsv}")
        return
        
    if not args.topics_file.exists():
        print(f"Error: Topics file không tồn tại: {args.topics_file}")
        return
    
    # Tạo thư mục output nếu cần
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input TSV: {args.input_tsv}")
    print(f"Topics file: {args.topics_file}")
    print(f"Output file: {args.output}")
    print("-" * 50)
    
    add_query_text_to_tsv(
        str(args.input_tsv), 
        str(args.topics_file), 
        str(args.output),
        quiet=bool(args.quiet),
        max_warnings=int(args.max_warnings),
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Sử dụng CLI nếu có arguments
        _cli()
    else:
        # Sử dụng interactive mode nếu không có arguments
        main()
