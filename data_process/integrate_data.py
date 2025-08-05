import os
import re

def parse_topics(topics_file_path):
    """
    Parses the topics file to extract query details.
    Each topic is expected to be enclosed in <top>...</top> tags.
    """
    topics_data = {}
    try:
        with open(topics_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Topics file not found at {topics_file_path}")
        return topics_data

    # Each topic is wrapped in <top>...</top>
    topic_entries = re.findall(r"<top>(.*?)</top>", content, re.DOTALL)
    for entry_content in topic_entries:
        num_match = re.search(r"<num>\s*Number:\s*(\d+)", entry_content)
        if not num_match:
            print(f"Warning: Found a topic entry without a number: {entry_content[:100]}...")
            continue
        num = num_match.group(1).strip()

        # Description: from <desc> up to </desc> or <narr>
        desc_match = re.search(r"<desc>\s*Description:(.*?)(?=<narr>|\Z)", entry_content, re.DOTALL)
        desc = desc_match.group(1).strip() if desc_match else ""

        # Narrative: from <narr> up to </narr> or end
        narr_match = re.search(r"<narr>\s*Narrative:(.*?)\Z", entry_content, re.DOTALL)
        narr = narr_match.group(1).strip() if narr_match else ""

        # Nối desc và narr với ". " như yêu cầu, bỏ qua title hoàn toàn, và làm sạch
        query_text = (desc + ". " + narr).replace('\n', ' ').replace('\r', '').replace('\t', ' ').strip()
        query_text = re.sub(r'\s+', ' ', query_text)  # Loại bỏ khoảng trắng thừa

        topics_data[num] = {'query_text': query_text}

    return topics_data

def main():
    workspace_root = "f:/SematicSearch/"  # Sửa đường dẫn nếu cần, ví dụ: "C:/Users/YourName/SemanticSearch/"
    qrels_file_path = os.path.join(workspace_root, "robustirdata/qrels.robust04.txt")
    topics_file_path = os.path.join(workspace_root, "robustirdata/topics.robust04.txt")
    data_dir = os.path.join(workspace_root, "robustirdata/data/")
    output_file_path = os.path.join(workspace_root, "integrated_robust04_modified_v2.tsv")  # Output mới để phân biệt

    print(f"Starting modified data integration script (version 2).")

    topics = parse_topics(topics_file_path)
    if not topics:
        print("No topics parsed or topics file not found. Exiting.")
        return

    output_lines_count = 0
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write("query_id\tquery_text\tdocument_id\tdocument\tlabel\n")  # Header

            with open(qrels_file_path, 'r', encoding='utf-8', errors='ignore') as qrels_file:
                for line_num, line in enumerate(qrels_file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 4:
                        print(f"Warning: Skipping malformed line {line_num}")
                        continue
                    query_id = parts[0].strip()
                    document_id = parts[2].strip()  # e.g., FBIS3-10082
                    label = parts[3].strip()

                    topic_info = topics.get(query_id)
                    if not topic_info:
                        print(f"Warning: No topic for query {query_id}. Skipping.")
                        continue

                    query_text = topic_info.get('query_text', '').replace('\t', ' ')
                    # Clean query text quotes for TSV consistency
                    query_text = query_text.replace('""', '"').replace('"', "'")  # Normalize quotes

                    actual_data_file_path = os.path.join(data_dir, document_id)
                    document = ""
                    try:
                        with open(actual_data_file_path, 'r', encoding='utf-8', errors='ignore') as df:
                            document = df.read()
                            # Clean document text for TSV format
                            document = document.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()
                            document = re.sub(r'\s+', ' ', document)  # Remove extra whitespace
                            
                            # Fix quotes for TSV format - normalize nested quotes and escape properly
                            # Replace double quotes with single quotes to avoid TSV parsing issues
                            document = document.replace('""', '"')  # Fix double-double quotes first
                            document = document.replace('"', "'")   # Convert all quotes to single quotes
                            
                    except FileNotFoundError:
                        document = "FILE_NOT_FOUND"
                    except Exception as e:
                        document = "ERROR_READING_FILE"

                    outfile.write(f"{query_id}\t{query_text}\t{document_id}\t{document}\t{label}\n")
                    output_lines_count += 1

    except Exception as e:
        print(f"Error: {e}")

    print(f"Finished. {output_lines_count} records written to {output_file_path}.")

if __name__ == "__main__":
    main()
