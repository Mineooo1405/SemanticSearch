import os
import re
import argparse
from pathlib import Path

def parse_topics(topics_file_path):
    """Trả về dict: {query_id: query_text(desc + ". " + narr)}."""
    topics_data = {}
    with open(topics_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    topic_entries = re.findall(r"<top>(.*?)</top>", content, re.DOTALL)
    for entry in topic_entries:
        num_match = re.search(r"<num>\s*Number:\s*(\d+)", entry)
        desc_match = re.search(r"<desc>\s*Description:(.*?)(?=<narr>|\Z)", entry, re.DOTALL)
        narr_match = re.search(r"<narr>\s*Narrative:(.*?)\Z", entry, re.DOTALL)
        if num_match:
            qid = num_match.group(1).strip()
            desc = desc_match.group(1).strip() if desc_match else ""
            narr = narr_match.group(1).strip() if narr_match else ""
            query_text = (desc + ". " + narr).replace('\n', ' ').replace('\r', '').replace('\t', ' ').strip()
            query_text = re.sub(r'\s+', ' ', query_text)
            topics_data[qid] = query_text
    return topics_data

def add_query_text_to_tsv(file_final_path: str, file_topics_path: str, output_path: str):
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

    print(f"Đã load {len(topics)} queries từ topics file")

    # Bước 2: Đọc file final, ghi file mới
    processed_lines = 0
    skipped_lines = 0
    
    with open(file_final_path, 'r', encoding='utf-8') as src, \
         open(output_path, 'w', encoding='utf-8') as out:
        
        header = src.readline()
        out.write("query_text\tchunk_text\tlabel\n")  # Header mới
        
        line_num = 1
        for line in src:
            line_num += 1
            items = line.rstrip('\n').split('\t')
            
            if len(items) != 3:
                print(f"Dòng lỗi format ở dòng {line_num}: {line.strip()}")
                skipped_lines += 1
                continue
                
            query_id, chunk_text, label = items
            query_text = topics.get(query_id, "")
            
            if not query_text:
                print(f"Không tìm thấy query_text với query_id={query_id} ở dòng {line_num}")
                skipped_lines += 1
                continue
                
            out.write(f"{query_text}\t{chunk_text}\t{label}\n")
            processed_lines += 1

    print(f"Đã xử lý: {processed_lines} dòng")
    print(f"Bỏ qua: {skipped_lines} dòng")
    print(f"Đã tạo file: {output_path}")
    return True


def main():
    file_final_path = input("Nhập đường dẫn đến file đã filter: ")
    file_topics_path = input("Nhập đường dẫn đến file topics: ")
    output_path = input("Đường dẫn file output [final_with_querytext.tsv]: ").strip() or "final_with_querytext.tsv"

    add_query_text_to_tsv(file_final_path, file_topics_path, output_path)


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
    
    success = add_query_text_to_tsv(
        str(args.input_tsv), 
        str(args.topics_file), 
        str(args.output)
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Sử dụng CLI nếu có arguments
        _cli()
    else:
        # Sử dụng interactive mode nếu không có arguments
        main()
