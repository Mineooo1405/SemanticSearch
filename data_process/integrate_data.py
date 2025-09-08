import os
import re
import hashlib
from collections import defaultdict

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
    # ---- Config (simple toggles; adjust as needed) ----
    DEDUP_BY_PAIR = True  # Bỏ trùng theo (query_id, document_id)
    DEDUP_SAME_CONTENT_WITHIN_QUERY = True  # Bỏ trùng khi cùng query_id và nội dung document sau clean giống nhau
    MIN_QUERY_LEN = 1  # Bỏ dòng nếu query_text (sau clean) ngắn hơn
    MIN_DOC_LEN = 1    # Bỏ dòng nếu document (sau clean) ngắn hơn
    SKIP_FILE_NOT_FOUND = True
    SKIP_ERROR_READING = True

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
    skipped_no_info_count = 0  # Track documents skipped for having no information
    skipped_empty_query = 0
    skipped_empty_doc = 0
    skipped_file_missing = 0
    skipped_read_error = 0
    skipped_dupe_pair = 0
    skipped_dupe_content = 0

    # Dedup trackers
    seen_pairs = set()  # (query_id, document_id)
    seen_doc_hash_by_query = defaultdict(set)  # query_id -> {md5(document_cleaned)}
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
                        # Missing topic mapping → cannot provide query_text
                        # Skip to maintain downstream 5-column integrity
                        continue

                    query_text = topic_info.get('query_text', '').replace('\t', ' ')
                    query_text = re.sub(r'\s+', ' ', query_text).strip()
                    # Clean query text quotes for TSV consistency
                    query_text = query_text.replace('""', '"').replace('"', "'")  # Normalize quotes

                    # Filter empty query
                    if len(query_text) < MIN_QUERY_LEN:
                        skipped_empty_query += 1
                        continue

                    # Dedup by (query_id, document_id)
                    if DEDUP_BY_PAIR:
                        key = (query_id, document_id)
                        if key in seen_pairs:
                            skipped_dupe_pair += 1
                            continue

                    actual_data_file_path = os.path.join(data_dir, document_id)
                    document = ""
                    try:
                        with open(actual_data_file_path, 'r', encoding='utf-8', errors='ignore') as df:
                            document = df.read()
                    except FileNotFoundError:
                        if SKIP_FILE_NOT_FOUND:
                            skipped_file_missing += 1
                            continue
                        document = "FILE_NOT_FOUND"
                    except Exception:
                        if SKIP_ERROR_READING:
                            skipped_read_error += 1
                            continue
                        document = "ERROR_READING_FILE"

                    # Clean document text for TSV format
                    document = document.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()
                    document = re.sub(r'\s+', ' ', document)  # Remove extra whitespace

                    # FILTER: Skip documents with no information
                    if document == "This document has no information.":
                        skipped_no_info_count += 1
                        continue

                    # Normalize quotes for TSV
                    document = document.replace('""', '"')  # Fix double-double quotes first
                    document = document.replace('"', "'")   # Convert all quotes to single quotes

                    # Filter empty/short doc
                    if len(document) < MIN_DOC_LEN:
                        skipped_empty_doc += 1
                        continue

                    # Dedup by same content within the same query
                    if DEDUP_SAME_CONTENT_WITHIN_QUERY:
                        doc_hash = hashlib.md5(document.encode('utf-8', errors='ignore')).hexdigest()
                        if doc_hash in seen_doc_hash_by_query[query_id]:
                            skipped_dupe_content += 1
                            continue

                    # Passed all filters → record and write
                    if DEDUP_BY_PAIR:
                        seen_pairs.add((query_id, document_id))
                    if DEDUP_SAME_CONTENT_WITHIN_QUERY:
                        seen_doc_hash_by_query[query_id].add(doc_hash)

                    outfile.write(f"{query_id}\t{query_text}\t{document_id}\t{document}\t{label}\n")
                    output_lines_count += 1

                    # Light progress
                    if output_lines_count % 200000 == 0:
                        print(f"Progress: written {output_lines_count:,} rows (line {line_num:,})")

    except Exception as e:
        print(f"Error: {e}")

    print(f"Finished. {output_lines_count} records written to {output_file_path}.")
    print(f"Skipped: no_info={skipped_no_info_count}, empty_query={skipped_empty_query}, empty_doc={skipped_empty_doc}, file_missing={skipped_file_missing}, read_error={skipped_read_error}, dupe_pair={skipped_dupe_pair}, dupe_content={skipped_dupe_content}")

if __name__ == "__main__":
    main()
