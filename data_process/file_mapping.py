import os
import re

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

def main():
    file_final_path = input("Nhập đường dẫn đến file đã filter: ")               # Đổi thành file của bạn
    file_topics_path = input("Nhập đường dẫn đến file topics: ")          # Đổi đường dẫn
    output_path = "final_with_querytext.tsv"                       # File đầu ra

    # Bước 1: Lấy dict mapping {query_id: query_text}
    topics = parse_topics(file_topics_path)
    if not topics:
        print("Không tìm thấy query_text từ topics. Kiểm tra lại file topics.")
        return

    # Bước 2: Đọc file final, ghi file mới
    with open(file_final_path, 'r', encoding='utf-8') as src, \
         open(output_path, 'w', encoding='utf-8') as out:
        header = src.readline()
        out.write("query_text\tchunk_text\tlabel\n")  # Header mới
        line_num = 1
        for line in src:
            line_num += 1
            items = line.rstrip('\n').split('\t')
            if len(items) != 3:
                print(f"Dòng lỗi format ở dòng {line_num}: {line}")
                continue
            query_id, chunk_text, label = items
            query_text = topics.get(query_id, "")
            if not query_text:
                print(f"Không tìm thấy query_text với query_id={query_id} ở dòng {line_num}")
                continue
            out.write(f"{query_text}\t{chunk_text}\t{label}\n")

    print(f"Đã tạo file: {output_path}")

if __name__ == "__main__":
    main()
