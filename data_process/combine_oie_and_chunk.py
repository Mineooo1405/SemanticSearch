import pandas as pd
import argparse
import os
import sys

def combine_and_create_3col(input_file_path, output_file_path):

    try:
        print(f"Reading input file: {input_file_path}")
        
        # Validate input file path
        if not input_file_path or not os.path.exists(input_file_path):
            print(f"Error: Input file '{input_file_path}' does not exist.")
            return False
            
        # Validate output file path
        if not output_file_path or output_file_path.strip() == "":
            print("Output file path not provided. Will auto-generate based on input file.")
        else:
            # Basic validation for provided output path
            pass
            
        # Read the input TSV file
        df = pd.read_csv(input_file_path, sep='\\t', engine='python')

        print("Input columns:", df.columns.tolist())

        # Ensure the required columns exist
        required_cols = ['query_text', 'chunk_text', 'raw_oie_data', 'label']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Input file must contain the following columns: {required_cols}")
            return False

        # Fill any NaN values in 'raw_oie_data' with an empty string
        df['raw_oie_data'] = df['raw_oie_data'].fillna('')

        # Combine 'chunk_text' and 'raw_oie_data'
        # Add a space between them if raw_oie_data is not empty
        df['passage'] = df.apply(
            lambda row: row['chunk_text'] + ' ' + row['raw_oie_data'] if pd.notna(row['raw_oie_data']) and row['raw_oie_data'].strip() != '' else row['chunk_text'],
            axis=1
        )

        # Create the new DataFrame with the desired columns
        output_df = pd.DataFrame({
            'query': df['query_text'],
            'passage': df['passage'],
            'label': df['label']
        })

        # If output_file_path is empty or not provided, create it based on input file
        if not output_file_path or output_file_path.strip() == "":
            input_dir = os.path.dirname(input_file_path)
            input_name = os.path.splitext(os.path.basename(input_file_path))[0]
            output_file_path = os.path.join(input_dir, f"{input_name}_3col.tsv")
            print(f"Auto-generated output path: {output_file_path}")
        else:
            # If only filename provided (no path), save in same directory as input
            if os.path.dirname(output_file_path) == "":
                input_dir = os.path.dirname(input_file_path)
                output_file_path = os.path.join(input_dir, output_file_path)
                print(f"Saving to input directory: {output_file_path}")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Save the new DataFrame to a TSV file
        output_df.to_csv(output_file_path, sep='\t', index=False, header=True)

        print(f"Successfully created 3-column TSV file at: {output_file_path}")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def interactive_mode():

    # Get input file path
    while True:
        input_file = input("Enter the path to the input TSV file: ").strip()
        if input_file and os.path.exists(input_file):
            break
        elif not input_file:
            print("Error: Please enter a file path.")
        else:
            print(f"Error: File '{input_file}' does not exist. Please try again.")
    
    # Get output file path
    while True:
        output_file = input("Enter the path for the output 3-column TSV file (or press Enter to auto-generate): ").strip()
        if output_file:
            # If user doesn't provide an extension, add .tsv
            if not output_file.endswith('.tsv'):
                output_file += '.tsv'
            break
        else:
            # If empty, will auto-generate based on input file
            output_file = ""
            print("Will auto-generate output filename based on input file.")
            break
    
    # Process the files
    success = combine_and_create_3col(input_file, output_file)
    
    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed.")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine text columns and create a 3-column TSV dataset.")
    parser.add_argument("input_file", nargs='?', help="Path to the input TSV file (6 or 8 columns).")
    parser.add_argument("output_file", nargs='?', help="Path for the output 3-column TSV file.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode.")

    args = parser.parse_args()

    # If no arguments provided or interactive flag is set, run interactive mode
    if args.interactive or (not args.input_file and not args.output_file):
        interactive_mode()
    else:
        if not args.input_file or not args.output_file:
            print("Error: Both input_file and output_file are required when not using interactive mode.")
            print("Use --interactive flag for interactive mode, or provide both file paths.")
            sys.exit(1)
        
        success = combine_and_create_3col(args.input_file, args.output_file)
        if not success:
            sys.exit(1)
