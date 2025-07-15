#!/usr/bin/env python3
# filepath: chunk_dataset.py

import os
import math
from pathlib import Path

def chunk_dataset_file():
    """
    Split TSV dataset file into n smaller parts
    """
    print("=" * 60)
    print("📊 DATASET CHUNKING TOOL")
    print("=" * 60)
    print("Tool to split TSV files into multiple smaller parts")
    print()
    
    # Input file path
    while True:
        input_file = input("📁 Enter path to TSV file: ").strip()
        if not input_file:
            print("❌ Please enter a file path!")
            continue
        
        if not os.path.exists(input_file):
            print(f"❌ File does not exist: {input_file}")
            continue
            
        if not input_file.lower().endswith('.tsv'):
            print("⚠️  Warning: File does not have .tsv extension")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                continue
        
        break
    
    # Input number of chunks
    while True:
        try:
            n_str = input("🔢 Enter number of chunks to create: ").strip()
            n = int(n_str)
            if n <= 0:
                print("❌ Number of chunks must be greater than 0!")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid integer!")
    
    # Read and analyze file
    print(f"\n📖 Reading file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # Read header
            header = f.readline()
            if not header.strip():
                print("❌ File is empty or has no header!")
                return
            
            # Read all data lines
            data_lines = f.readlines()
            
        total_rows = len(data_lines)
        print(f"📊 Total data rows: {total_rows}")
        print(f"📋 Header: {header.strip()}")
        
        if total_rows == 0:
            print("❌ File has no data!")
            return
            
        if n > total_rows:
            print(f"⚠️  Warning: Number of chunks ({n}) is greater than data rows ({total_rows})")
            print("Some chunk files will be empty.")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Calculate chunk size
    chunk_size = math.ceil(total_rows / n)
    print(f"🔢 Size per chunk: ~{chunk_size} rows")
    
    # Create base filename
    file_path = Path(input_file)
    base_name = file_path.stem  # Filename without extension
    output_dir = file_path.parent
    
    print(f"\n🚀 Starting to split file into {n} parts...")
    print("-" * 50)
    
    # Create chunk files
    created_files = []
    
    for i in range(n):
        chunk_name = output_dir / f"{base_name}_chunk{i+1}.tsv"
        
        # Calculate row indices for this chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        
        # Get data for this chunk
        chunk_data = data_lines[start_idx:end_idx]
        
        try:
            with open(chunk_name, 'w', encoding='utf-8') as chunk_file:
                # Write header
                chunk_file.write(header)
                
                # Write data
                for line in chunk_data:
                    chunk_file.write(line)
            
            created_files.append(chunk_name)
            
            # Display chunk information
            actual_rows = len(chunk_data)
            print(f"✅ Chunk {i+1:2d}: {chunk_name.name} ({actual_rows:4d} rows)")
            
        except Exception as e:
            print(f"❌ Error creating chunk {i+1}: {e}")
            return
    
    # Summary
    print("-" * 50)
    print(f"🎉 Complete! Split into {len(created_files)} files:")
    print()
    
    total_output_rows = 0
    for i, file_path in enumerate(created_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                data_rows = len(lines) - 1  # Subtract header
                total_output_rows += data_rows
                file_size = os.path.getsize(file_path)
                print(f"📄 {file_path.name}: {data_rows} rows ({file_size:,} bytes)")
        except:
            print(f"📄 {file_path.name}: Cannot read")
    
    print()
    print(f"📊 Summary:")
    print(f"   Input:  {total_rows:,} rows")
    print(f"   Output: {total_output_rows:,} rows")
    print(f"   Files:  {len(created_files)} chunks")
    
    if total_rows == total_output_rows:
        print("✅ File split successfully - no data loss!")
    else:
        print(f"⚠️  Warning: Row count difference detected!")

def main():
    """Main function"""
    try:
        chunk_dataset_file()
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()