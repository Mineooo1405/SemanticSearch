#!/usr/bin/env python3
"""
Convert 6-column TSV to 3-column TSV for chunking
Input format: number | title | Description | Narrative | Label | Data
Output format: query | passage | label

Query = title + ". " + description + ". " + narrative
Passage = data
Label = label
"""

import csv
import sys
import os
from pathlib import Path
from typing import Optional

# Tăng giới hạn kích thước trường để xử lý các passage rất dài
csv.field_size_limit(min(sys.maxsize, 2147483647))

def convert_tsv_format(input_file: str, output_file: Optional[str] = None, *,
                       keep_unprocessed: bool = True) -> str:
    """
    Convert TSV từ nhiều định dạng (6 hoặc 8 cột) về định dạng 3 cột

    Hỗ trợ:
        • 6 cột  (number | title | description | narrative | label | data)
        • 8 cột  (query_id | query_text | passage_id | original_passage | label | chunk_id | chunk_text | raw_oie_data)

    Kết quả luôn có dạng: query    passage    label

    Args:
        input_file: Đường dẫn file TSV gốc (6 hoặc 8 cột)
        output_file: Đường dẫn file TSV đầu ra (3 cột). Nếu None sẽ tự sinh.

    Returns:
        Đường dẫn file đầu ra
    """
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Auto-generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_3col{input_path.suffix}")
    
    print(f"Converting TSV format...")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    
    rows_processed = 0
    rows_written = 0
    
    skipped_rows: list[list[str]] = []

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')
            
            # Read and validate header
            try:
                header = next(reader)
                
                # Xác định số cột và kiểu định dạng
                if len(header) >= 8 and {'query_text', 'chunk_text'}.issubset(set([h.lower() for h in header])):
                    input_format = '8col'
                elif len(header) >= 6:
                    input_format = '6col'
                elif len(header) >= 7 and {'query_text', 'chunk_text'}.issubset(set([h.lower() for h in header])):
                    input_format = '7col'
                else:
                    raise ValueError(f"Unsupported column count ({len(header)}) in header: {header}")
                
                print(f"Input header ({len(header)} columns): {header}")
                
                # Write new header
                new_header = ['query', 'passage', 'label']
                writer.writerow(new_header)
                print(f"Output header ({len(new_header)} columns): {new_header}")
                
            except StopIteration:
                raise ValueError("Input file is empty or has no header")
            
            # Process data rows
            for row_idx, row in enumerate(reader, 1):
                rows_processed += 1
                
                try:
                    required_cols = 8 if input_format == '8col' else 6

                    if len(row) < required_cols:
                        print(f"Skipping row {row_idx}: insufficient columns ({len(row)}/{required_cols})")
                        if keep_unprocessed:
                            skipped_rows.append(row)
                        continue

                    if input_format == '6col':
                        # number | title | description | narrative | label | data
                        _, title, description, narrative, label, data = row[:6]

                        # Ghép title + description + narrative
                        query_parts = [part.strip() for part in (title, description, narrative) if part.strip()]
                        query = '. '.join(query_parts)
                        passage = data.strip()
                    
                    elif input_format == '7col':
                        #query_id | query_text | passage_id | label | chunk_id | chunk_text
                        query_text = row[1].strip()
                        chunk_text = row[5].strip()
                        label = row[3].strip()
                        
                        query = query_text
                        passage = chunk_text
                        

                    else:  # 8col
                        # query_id | query_text | passage_id | original_passage | label | chunk_id | chunk_text | raw_oie_data
                        query_text = row[1].strip()
                        chunk_text = row[6].strip()
                        label = row[4].strip()

                        query = query_text
                        passage = chunk_text
                    
                    # Validate data
                    if not query:
                        print(f"Skipping row {row_idx}: empty query")
                        if keep_unprocessed:
                            skipped_rows.append(row)
                        continue
                    
                    if not passage:
                        print(f"Skipping row {row_idx}: empty passage")
                        if keep_unprocessed:
                            skipped_rows.append(row)
                        continue
                    
                    if not label:
                        print(f"Skipping row {row_idx}: empty label")
                        if keep_unprocessed:
                            skipped_rows.append(row)
                        continue
                    
                    # Write converted row
                    writer.writerow([query, passage, label])
                    rows_written += 1
                    
                    # Progress indicator
                    if rows_processed % 100 == 0:
                        print(f"Processed {rows_processed} rows, written {rows_written} rows...")
                
                except Exception as e:
                    print(f"Error processing row {row_idx}: {e}")
                    if keep_unprocessed:
                        skipped_rows.append(row)
                    continue
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
    
    print(f"\nConversion completed!")
    print(f"Rows processed: {rows_processed}")
    print(f"Rows written: {rows_written}")
    print(f"Output file: {output_file}")
    
    # Show file size
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size:,} bytes")

    # --- Rewrite input file để chỉ còn hàng chưa xử lý ---
    if keep_unprocessed:
        try:
            if skipped_rows:
                tmp_path = Path(input_file).with_suffix(".remaining.tsv.tmp")
                with open(tmp_path, "w", encoding="utf-8", newline="") as tf:
                    writer = csv.writer(tf, delimiter="\t")
                    writer.writerow(header)  # original header
                    writer.writerows(skipped_rows)
                os.replace(tmp_path, input_file)
                print(f"Input file cập nhật: còn {len(skipped_rows)} hàng chưa xử lý")
            else:
                # Không còn hàng nào → xóa file nguồn
                os.remove(input_file)
                print("Đã xử lý hết. Đã xóa file nguồn ban đầu.")
        except Exception as e:
            print(f"Warning: Không thể cập nhật file nguồn: {e}")
    
    return output_file

# ------------------------------------------------------
# Utility: đếm tổng số dòng (không tính header)
# ------------------------------------------------------


def count_rows(file_path: str) -> int:
    """Đếm số hàng dữ liệu (bỏ qua header) trong TSV."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1

def preview_file(file_path: str, num_rows: int = 3):
    """Preview first few rows of the file"""
    print(f"\nPreview of {os.path.basename(file_path)} (first {num_rows} rows):")
    print("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i >= num_rows + 1:  # +1 for header
                    break
                
                if i == 0:
                    print(f"Header: {row}")
                    print("-" * 80)
                else:
                    print(f"Row {i}:")
                    for j, col in enumerate(row):
                        col_preview = col[:100] + "..." if len(col) > 100 else col
                        print(f"  Column {j+1}: {col_preview}")
                    print()
    
    except Exception as e:
        print(f"Error previewing file: {e}")

def main():
    """Main interactive function"""
    print("TSV FORMAT CONVERTER")
    print("=" * 50)
    print("Converts 6-column TSV to 3-column TSV for chunking")
    print("Input:  number | title | Description | Narrative | Label | Data")
    print("Output: query | passage | label")
    print()
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Enter path to input TSV file: ").strip()
    
    if not input_file:
        print("No input file specified!")
        return
    
    # Get output file
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_choice = input("\nSpecify output file? (y/n, default=auto): ").lower()
        if output_choice == 'y':
            output_file = input("Enter output file path: ").strip()
    
    try:
        # Convert file
        result_file = convert_tsv_format(input_file, output_file)
        
        # Ask for preview
        preview_choice = input(f"\nPreview output file? (y/n, default=y): ").lower()
        if preview_choice != 'n':
            preview_file(result_file)
        
        print(f"\nSUCCESS! 3-column TSV ready for chunking:")
        print(f"{result_file}")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        return

if __name__ == "__main__":
    main()
