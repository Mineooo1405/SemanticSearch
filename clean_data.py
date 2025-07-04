import sys
import csv

csv.field_size_limit(min(sys.maxsize, 2147483647))

def escape_csv_field(field_text):
    """
    Properly escape a field for CSV by replacing internal quotes and wrapping in quotes.
    """
    # Replace any existing quotes with escaped quotes
    escaped_text = field_text.replace('"', '""')
    # Wrap the entire field in quotes
    return f'"{escaped_text}"'

def clean_csv_data(input_filename, output_filename):
    """
    Read CSV file and write to new file with comma delimiter.
    No cleaning of brackets is performed - data is kept as-is.
    """
    rows_processed = 0
    
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
             open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile, delimiter='|')
            
            # Process header
            header = next(reader, None)
            if header:
                # Write header manually with commas
                header_line = ','.join(header) + '\n'
                outfile.write(header_line)
                print(f"Header: {' | '.join(header)}")
                print("Note: Output file will use comma (,) as delimiter")
            
            # Process data rows
            for row in reader:
                rows_processed += 1
                
                if len(row) > 0:
                    # Keep all data as-is, no cleaning
                    data_text = row[-1]  # Last column (Data column)
                    
                    # Prepare the row for output
                    output_row = row[:-1]  # All columns except the last one
                    
                    # Write the row manually
                    row_parts = []
                    for field in output_row:
                        row_parts.append(field)
                    
                    # Add the data column with proper quotes (no cleaning)
                    row_parts.append(escape_csv_field(data_text))
                    
                    # Write the complete row
                    row_line = ','.join(row_parts) + '\n'
                    outfile.write(row_line)
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return False
    except Exception as e:
        print(f"Error processing file: {e}")
        return False
    
    print(f"\nProcessing complete!")
    print(f"Total rows processed: {rows_processed}")
    print(f"Data saved to: {output_filename}")
    print("Note: No bracket cleaning was performed - all data kept as-is")
    
    return True

def main():
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = input("Enter input CSV file path: ").strip()
        output_file = input("Enter output CSV file path: ").strip()
        
        if not output_file:
            if input_file.endswith('.csv'):
                output_file = input_file[:-4] + '_converted.csv'
            else:
                output_file = input_file + '_converted.csv'
            print(f"Using default output filename: {output_file}")
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("Starting data conversion...")
    
    success = clean_csv_data(input_file, output_file)
    
    if success:
        print("\nData conversion completed successfully!")
    else:
        print("\nData conversion failed.")

if __name__ == "__main__":
    main()
