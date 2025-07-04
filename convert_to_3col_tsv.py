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

def convert_tsv_format(input_file: str, output_file: str = None) -> str:
    """
    Convert 6-column TSV to 3-column TSV
    
    Args:
        input_file: Path to input TSV file (6 columns)
        output_file: Path to output TSV file (3 columns). If None, auto-generate
    
    Returns:
        Path to output file
    """
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Auto-generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_3col{input_path.suffix}")
    
    print(f"🔄 Converting TSV format...")
    print(f"📂 Input:  {input_file}")
    print(f"📁 Output: {output_file}")
    
    rows_processed = 0
    rows_written = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')
            
            # Read and validate header
            try:
                header = next(reader)
                if len(header) < 6:
                    raise ValueError(f"Expected 6 columns, got {len(header)}: {header}")
                
                print(f"📋 Input header ({len(header)} columns): {header}")
                
                # Write new header
                new_header = ['query', 'passage', 'label']
                writer.writerow(new_header)
                print(f"📝 Output header ({len(new_header)} columns): {new_header}")
                
            except StopIteration:
                raise ValueError("Input file is empty or has no header")
            
            # Process data rows
            for row_idx, row in enumerate(reader, 1):
                rows_processed += 1
                
                try:
                    if len(row) < 6:
                        print(f"⚠️  Skipping row {row_idx}: insufficient columns ({len(row)}/6)")
                        continue
                    
                    # Extract columns: number, title, description, narrative, label, data
                    number, title, description, narrative, label, data = row[0], row[1], row[2], row[3], row[4], row[5]
                    
                    # Clean and combine query components
                    title = title.strip()
                    description = description.strip()
                    narrative = narrative.strip()
                    
                    # Create query by combining title, description, and narrative
                    query_parts = []
                    if title:
                        query_parts.append(title)
                    if description:
                        query_parts.append(description)
                    if narrative:
                        query_parts.append(narrative)
                    
                    query = ". ".join(query_parts).strip()
                    if query.endswith('.'):
                        query = query[:-1]  # Remove trailing dot if it ends with one
                    
                    # Clean passage (data)
                    passage = data.strip()
                    
                    # Clean label
                    label = label.strip()
                    
                    # Validate data
                    if not query:
                        print(f"⚠️  Skipping row {row_idx}: empty query")
                        continue
                    
                    if not passage:
                        print(f"⚠️  Skipping row {row_idx}: empty passage")
                        continue
                    
                    if not label:
                        print(f"⚠️  Skipping row {row_idx}: empty label")
                        continue
                    
                    # Write converted row
                    writer.writerow([query, passage, label])
                    rows_written += 1
                    
                    # Progress indicator
                    if rows_processed % 100 == 0:
                        print(f"📊 Processed {rows_processed} rows, written {rows_written} rows...")
                
                except Exception as e:
                    print(f"❌ Error processing row {row_idx}: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        raise
    
    print(f"\n✅ Conversion completed!")
    print(f"📊 Rows processed: {rows_processed}")
    print(f"📝 Rows written: {rows_written}")
    print(f"📁 Output file: {output_file}")
    
    # Show file size
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"📦 File size: {file_size:,} bytes")
    
    return output_file

def preview_file(file_path: str, num_rows: int = 3):
    """Preview first few rows of the file"""
    print(f"\n🔍 Preview of {os.path.basename(file_path)} (first {num_rows} rows):")
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
        print(f"❌ Error previewing file: {e}")

def main():
    """Main interactive function"""
    print("🔄 TSV FORMAT CONVERTER")
    print("=" * 50)
    print("Converts 6-column TSV to 3-column TSV for chunking")
    print("Input:  number | title | Description | Narrative | Label | Data")
    print("Output: query | passage | label")
    print()
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("📂 Enter path to input TSV file: ").strip()
    
    if not input_file:
        print("❌ No input file specified!")
        return
    
    # Get output file
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_choice = input("\n📁 Specify output file? (y/n, default=auto): ").lower()
        if output_choice == 'y':
            output_file = input("📁 Enter output file path: ").strip()
    
    try:
        # Convert file
        result_file = convert_tsv_format(input_file, output_file)
        
        # Ask for preview
        preview_choice = input(f"\n🔍 Preview output file? (y/n, default=y): ").lower()
        if preview_choice != 'n':
            preview_file(result_file)
        
        print(f"\n🎉 SUCCESS! 3-column TSV ready for chunking:")
        print(f"📁 {result_file}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return

if __name__ == "__main__":
    main()
