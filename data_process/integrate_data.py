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

    # Each topic is wrapped in <top> ... </top>
    # re.DOTALL allows . to match newlines
    topic_entries = re.findall(r"<top>(.*?)</top>", content, re.DOTALL)

    for entry_content in topic_entries:
        num_match = re.search(r"<num>\s*Number:\s*(\d+)", entry_content)
        if not num_match:
            print(f"Warning: Found a topic entry without a number: {entry_content[:100]}...")
            continue 
        
        num = num_match.group(1).strip()
        
        # Title: from <title> up to <desc>, <narr>, or end of entry_content
        title_match = re.search(r"<title>(.*?)(?=<desc>|<narr>|\Z)", entry_content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        # Description: from <desc> Description: up to <narr> or end of entry_content
        desc_match = re.search(r"<desc>\s*Description:(.*?)(?=<narr>|\Z)", entry_content, re.DOTALL)
        desc = desc_match.group(1).strip() if desc_match else ""
        
        # Narrative: from <narr> Narrative: up to end of entry_content
        narr_match = re.search(r"<narr>\s*Narrative:(.*?)\Z", entry_content, re.DOTALL)
        narr = narr_match.group(1).strip() if narr_match else ""
        
        topics_data[num] = {
            'title': title.replace('\n', ' ').replace('\r', '').strip(),
            'desc': desc.replace('\n', ' ').replace('\r', '').strip(),
            'narr': narr.replace('\n', ' ').replace('\r', '').strip()
        }
        
    return topics_data

def main():
    workspace_root = "f:/SematicSearch/"
    qrels_file_path = os.path.join(workspace_root, "robustirdata/qrels.robust04.txt")
    topics_file_path = os.path.join(workspace_root, "robustirdata/topics.robust04.txt") # Corrected typo from potential "roboust04"
    data_dir = os.path.join(workspace_root, "robustirdata/data/")
    output_file_path = os.path.join(workspace_root, "integrated_robust04_data.tsv")

    print(f"Starting data integration script.")
    print(f"Parsing topics from: {topics_file_path}")
    topics = parse_topics(topics_file_path)
    if not topics:
        print("No topics parsed or topics file not found. Exiting.")
        return

    print(f"Processing qrels file: {qrels_file_path}")
    print(f"Data directory: {data_dir}")
    print(f"Writing output to: {output_file_path}")

    output_lines_count = 0
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write("number\ttitle\tDescription\tNarrative\tLabel\tData\n") # Header

            try:
                with open(qrels_file_path, 'r', encoding='utf-8', errors='ignore') as qrels_file:
                    for line_num, line in enumerate(qrels_file, 1):
                        line = line.strip()
                        if not line: # Skip empty lines
                            continue

                        parts = line.split() # Splits by whitespace
                        if len(parts) != 4:
                            print(f"Warning: Skipping malformed line {line_num} in qrels (expected 4 parts, got {len(parts)}): '{line}'")
                            continue
                        
                        query_num = parts[0].strip()
                        doc_id_from_qrels = parts[2].strip() # e.g., FBIS3-10082
                        label = parts[3].strip()

                        topic_info = topics.get(query_num)
                        if not topic_info:
                            print(f"Warning: No topic information found for query number {query_num} (from qrels line {line_num}). Skipping this entry.")
                            continue

                        title = topic_info.get('title', '').replace('\t', ' ') # Replace tabs with spaces
                        description = topic_info.get('desc', '').replace('\t', ' ') # Replace tabs with spaces
                        narrative = topic_info.get('narr', '').replace('\t', ' ') # Replace tabs with spaces
                        
                        data_file_name = doc_id_from_qrels # Use doc_id directly
                        actual_data_file_path = os.path.join(data_dir, data_file_name)
                        
                        data_content = ""
                        try:
                            with open(actual_data_file_path, 'r', encoding='utf-8', errors='ignore') as df:
                                data_content = df.read()
                            # Normalize data: replace newlines with spaces, and tabs with spaces
                            data_content = data_content.replace('\t', ' ').replace('\n', ' ').replace('\r', '').strip()
                        except FileNotFoundError:
                            # print(f"Warning: Data file not found: {actual_data_file_path} (referenced in qrels line {line_num}). Data column will be 'FILE_NOT_FOUND'.")
                            data_content = "FILE_NOT_FOUND"
                        except Exception as e:
                            print(f"Error reading data file {actual_data_file_path} (qrels line {line_num}): {e}. Data column will be 'ERROR_READING_FILE'.")
                            data_content = "ERROR_READING_FILE"
                        
                        outfile.write(f"{query_num}\t{title}\t{description}\t{narrative}\t{label}\t{data_content}\n")
                        output_lines_count += 1
            
            except FileNotFoundError:
                print(f"Error: Qrels file not found at {qrels_file_path}. Output file may be incomplete or empty (only header).")
                return 
            except Exception as e:
                print(f"An error occurred during qrels file processing: {e}")
                return

    except IOError as e:
        print(f"Fatal Error: Could not write to output file {output_file_path}: {e}")
        return

    print(f"Script finished. {output_lines_count} data records written to {output_file_path}.")

if __name__ == "__main__":
    main()
