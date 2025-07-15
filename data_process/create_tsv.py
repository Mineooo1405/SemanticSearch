import sys
import csv
import os
import re

csv.field_size_limit(min(sys.maxsize, 2147483647))

def detect_delimiter(file_path, sample_lines=5):
    """Detect the delimiter used in CSV file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = ""
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                sample += line

        tab_count = sample.count('\t')
        pipe_count = sample.count('|')
        comma_count = sample.count(',')
        semicolon_count = sample.count(';')
        
        print(f"Delimiter detection:")
        print(f"  Tab count: {tab_count}")
        print(f"  Pipe count: {pipe_count}")
        print(f"  Comma count: {comma_count}")
        print(f"  Semicolon count: {semicolon_count}")
        
        # For .tsv files, prioritize tab if it exists
        if file_path.lower().endswith('.tsv') and tab_count > 0:
            return '\t'
        
        # Find the most frequent delimiter
        delimiter_counts = {
            '|': pipe_count,
            ',': comma_count,
            '\t': tab_count,
            ';': semicolon_count
        }
        
        # Get delimiter with highest count
        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        best_count = delimiter_counts[best_delimiter]
        
        print(f"Best delimiter: '{best_delimiter}' (count: {best_count})")

        if best_count > 0:
            test_lines = sample.strip().split('\n')
            if len(test_lines) > 0:
                test_columns = test_lines[0].split(best_delimiter)
                print(f"Test split with '{best_delimiter}': {len(test_columns)} columns")

                if len(test_columns) >= 6:
                    print(f"Delimiter '{best_delimiter}' gives reasonable column count")
                    return best_delimiter
        
        # Fallback logic
        if pipe_count > comma_count and pipe_count > tab_count:
            return '|'
        elif tab_count > comma_count:
            return '\t'
        else:
            return ','
            
    except Exception as e:
        print(f"Error detecting delimiter: {e}")
        return ','  

def validate_tsv_format(file_path):
    """Validate if TSV file has proper format"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            first_row = next(reader, None)
            
            if not header:
                return False, "No header found"
            
            if len(header) < 3:
                return False, f"Header has {len(header)} columns, expected 3"
            
            if first_row and len(first_row) < 3:
                return False, f"First data row has {len(first_row)} columns, expected 3"
            
            return True, "Valid TSV format"
            
    except Exception as e:
        return False, f"Error validating: {e}"

def fix_tsv_header(file_path):
    
    print(f"Fixing TSV header in: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("Error: File is empty")
            return None
        
        # Show original header
        original_header = lines[0].strip()
        print(f"Original header: {repr(original_header)}")

        header_parts = original_header.split('\t')
        if len(header_parts) == 3:
            print("Header already has 3 tab-separated parts, no fix needed")
            return file_path
        
        # Try to fix header with common patterns
        # Pattern 1: "query   passage	label" (spaces + tab)
        if 'query' in original_header and 'passage' in original_header and 'label' in original_header:
            lines[0] = "query\tpassage\tlabel\n"
            print("Fixed header to: query\\tpassage\\tlabel")
        else:
            print("Warning: Header doesn't contain expected column names")
            return None
        
        # Create fixed file
        output_path = file_path.replace('.tsv', '_fixed.tsv')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"Fixed file saved to: {output_path}")
        
        # Validate fixed file
        is_valid, message = validate_tsv_format(output_path)
        if is_valid:
            print("Fixed file has valid format!")
        else:
            print(f"Fixed file still has issues: {message}")
            return None
        
        return output_path
        
    except Exception as e:
        print(f"Error fixing header: {e}")
        return None

def create_tsv_file(input_filename, output_filename, query_type="title_description", delimiter=None):

    rows_processed = 0
    
    try:
        # Auto-detect delimiter if not provided
        if delimiter is None:
            delimiter = detect_delimiter(input_filename)
            print(f"Auto-detected delimiter: '{delimiter}' ({'TAB' if delimiter == '\t' else delimiter})")
        
        with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
             open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile, delimiter=delimiter)
            
            # Write TSV header
            outfile.write("query\tpassage\tlabel\n")
            
            # Skip header
            header = next(reader, None)
            if header:
                print(f"Header found ({len(header)} columns): {[col.strip() for col in header]}")
                print(f"Delimiter used: '{delimiter}' ({'TAB' if delimiter == '\t' else delimiter})")
                print(f"Query type: {query_type}")
                print("Processing rows...")
            
            # Process data rows
            for row in reader:
                rows_processed += 1
                
                if len(row) >= 6:  # Ensure we have all required columns
                    number = row[0].strip()
                    title = row[1].strip()
                    description = row[2].strip()
                    narrative = row[3].strip()
                    label = row[4].strip()
                    data = row[5].strip()
                    
                    # Skip if essential data is missing
                    if not data.strip():
                        print(f"Warning: Row {rows_processed} has empty data field, skipping")
                        continue
                    
                    # Create query based on type
                    if query_type == "title_description":
                        query = f"{title}. {description}"
                    elif query_type == "description":
                        query = description
                    elif query_type == "title":
                        query = title
                    else:
                        query = f"{title}. {description}"  # Default
                    
                    # Clean query and passage (remove tabs and newlines that could break TSV)
                    query = query.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()
                    passage = data.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()
                    label = label.strip()
                    
                    # Write TSV line: query \t passage \t label
                    tsv_line = f"{query}\t{passage}\t{label}\n"
                    outfile.write(tsv_line)
                    
                    # Show progress every 1000 rows
                    if rows_processed % 1000 == 0:
                        print(f"Processed {rows_processed} rows...")
                        
                    # Show first few examples
                    if rows_processed <= 3:
                        print(f"\nExample {rows_processed}:")
                        print(f"Query: {query[:100]}...")
                        print(f"Passage: {passage[:100]}...")
                        print(f"Label: {label}")
                        print("-" * 50)
                else:
                    print(f"Warning: Row {rows_processed} has insufficient columns ({len(row)}), expected 6")
                    if len(row) > 0:
                        print(f"Row content: {row[:3]}...")  # Show first few fields
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return False
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\nProcessing complete!")
    print(f"Total rows processed: {rows_processed}")
    print(f"TSV file saved to: {output_filename}")
    print(f"Format: query \\t passage \\t label")
    
    # Validate output file
    try:
        with open(output_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            data_lines = len(lines) - 1  # Exclude header
            print(f"Output TSV: {data_lines} data rows + 1 header = {len(lines)} total lines")
            
            if len(lines) > 1:
                # Validate first data line
                first_data_line = lines[1].strip().split('\t')
                print(f"First data row has {len(first_data_line)} fields (expected: 3)")
                
                if len(first_data_line) != 3:
                    print(f"Warning: Expected 3 TSV fields but got {len(first_data_line)}")
                
    except Exception as e:
        print(f"Warning: Could not validate output file: {e}")
    
    return True

def get_delimiter_choice():
    """Ask user to choose delimiter"""
    print("\nDelimiter options:")
    print("1. Pipe (|) - Common in research datasets")
    print("2. Comma (,) - Standard CSV format")
    print("3. Tab (\\t) - TSV format")
    print("4. Auto-detect from file")
    
    while True:
        choice = input("Choose delimiter (1-4, default=4): ").strip()
        
        if choice == "1":
            return "|"
        elif choice == "2":
            return ","
        elif choice == "3":
            return "\t"
        elif choice == "4" or choice == "":
            return None  # Auto-detect
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def analyze_file_structure(file_path):
    """Analyze file structure to understand format better"""
    print(f"Analyzing file structure: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]  # First 10 lines
        
        print(f"Total sample lines: {len(lines)}")
        
        # Test different delimiters
        delimiters = ['\t', '|', ',', ';']
        
        for delimiter in delimiters:
            print(f"\n--- Testing delimiter: '{delimiter}' ---")
            
            for i, line in enumerate(lines[:3]):  # First 3 lines
                line = line.strip()
                if line:
                    parts = line.split(delimiter)
                    print(f"Line {i+1}: {len(parts)} parts")
                    if len(parts) <= 10:  # Only show if reasonable number
                        for j, part in enumerate(parts):
                            part_preview = part[:50] + "..." if len(part) > 50 else part
                            print(f"  [{j}] {part_preview}")
                    else:
                        print(f"  Too many parts ({len(parts)}) - likely wrong delimiter")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

def main():
    print("TSV File Creator/Fixer for Neural Ranking Models")
    print("=" * 60)
    
    # Get input parameters
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        query_type = sys.argv[3] if len(sys.argv) > 3 else "title_description"
        delimiter = sys.argv[4] if len(sys.argv) > 4 else None
    else:
        input_file = input("Enter input file path: ").strip().replace('"', '')
        
        if not input_file:
            print("Error: Input file path is required!")
            return
        
        # Check if input file exists and is valid
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found!")
            return
        
        if os.path.isdir(input_file):
            print(f"Error: '{input_file}' is a directory, not a file!")
            return
        
        # Analyze file structure before processing
        analyze_choice = input("Analyze file structure first? (y/n, default=y): ").strip().lower()
        if analyze_choice != 'n':
            analyze_file_structure(input_file)
            
            manual_delimiter = input("\nSpecify delimiter manually? (|,tab,comma,auto, default=auto): ").strip().lower()
            if manual_delimiter == '|':
                delimiter = '|'
                print("Using pipe delimiter: |")
            elif manual_delimiter == 'tab':
                delimiter = '\t'
                print("Using tab delimiter")
            elif manual_delimiter == 'comma':
                delimiter = ','
                print("Using comma delimiter")
            else:
                delimiter = None
                print("Will auto-detect delimiter")
        
        # Special handling for TSV files
        if input_file.lower().endswith('.tsv'):
            print(f"\nDetected TSV file: {input_file}")
            
            # Validate current format
            is_valid, message = validate_tsv_format(input_file)
            print(f"Format validation: {message}")
            
            if not is_valid:
                print("\n🔧 TSV file has format issues. Options:")
                print("1. Fix TSV header and validate")
                print("2. Convert from CSV to TSV (full processing)")
                print("3. Exit")
                
                choice = input("Choose option (1-3, default=1): ").strip()
                
                if choice == "1" or choice == "":
                    # Fix TSV header
                    fixed_file = fix_tsv_header(input_file)
                    if fixed_file:
                        print(f"\nUse this fixed file:")
                        print(f"   {fixed_file}")
                        
                        use_fixed = input("\nUse fixed file for further processing? (y/n, default=n): ").strip().lower()
                        if use_fixed == 'y':
                            input_file = fixed_file
                        else:
                            print("Header fix complete. You can now use the fixed file.")
                            return
                    else:
                        print("Could not fix header automatically")
                        return
                elif choice == "3":
                    return
                # choice == "2" continues to full processing
            else:
                print("TSV file format is valid!")
                
                # Offer to just copy/validate
                copy_choice = input("File is already valid TSV. Just copy/validate? (y/n, default=y): ").strip().lower()
                if copy_choice != 'n':
                    output_file = input("Enter output file path (press Enter for auto): ").strip()
                    if not output_file:
                        output_file = input_file.replace('.tsv', '_validated.tsv')
                        print(f"Using: {output_file}")
                    
                    try:
                        import shutil
                        shutil.copy2(input_file, output_file)
                        print(f"File copied to: {output_file}")
                        
                        # Quick validation
                        with open(output_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            print(f"Total lines: {len(lines)}")
                            if len(lines) > 1:
                                first_data = lines[1].strip().split('\t')
                                print(f"First data row: {len(first_data)} fields")
                        return
                    except Exception as e:
                        print(f"Error copying file: {e}")
                        return
        
        # Get output file path
        output_file = input("Enter output TSV file path (press Enter for auto): ").strip()
        
        # Auto-generate output filename if not provided
        if not output_file:
            if input_file.endswith('.csv'):
                output_file = input_file[:-4] + '_converted.tsv'
            elif input_file.endswith('.tsv'):
                output_file = input_file[:-4] + '_processed.tsv'
            else:
                output_file = input_file + '_converted.tsv'
            print(f"Using auto-generated output filename: {output_file}")
        
        # Get delimiter choice for CSV files (if not already set)
        if delimiter is None:
            delimiter = get_delimiter_choice()
        
        # Get query type for CSV conversion
        print("\nQuery type options:")
        print("1. title_description (recommended) - Combines title and description")
        print("2. description - Uses description only")
        print("3. title - Uses title only")
        choice = input("Choose query type (1-3, default=1): ").strip()
        
        query_type_map = {"1": "title_description", "2": "description", "3": "title"}
        query_type = query_type_map.get(choice, "title_description")
    
    # Convert delimiter string to actual character if provided as string
    if delimiter == "pipe":
        delimiter = "|"
    elif delimiter == "comma":
        delimiter = ","
    elif delimiter == "tab":
        delimiter = "\t"
    
    # Check output path validity
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist!")
        return
    
    if os.path.isdir(output_file):
        print(f"Error: '{output_file}' is a directory, not a file!")
        return
    
    print(f"\nConfiguration:")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Query type: {query_type}")
    print(f"Delimiter: {'Auto-detect' if delimiter is None else repr(delimiter)}")
    print("Starting processing...")
    
    success = create_tsv_file(input_file, output_file, query_type, delimiter)
    
    if success:
        print("\nTSV file created successfully!")
        print("You can now use this file for your semantic search project.")
        print(f"\nNext step - Run controller:")
        print(f"python data_create_controller.py")
        print(f"Input file: {output_file}")
    else:
        print("\nProcessing failed.")

if __name__ == "__main__":
    main()