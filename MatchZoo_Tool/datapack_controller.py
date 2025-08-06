# -*- coding: utf-8 -*-
"""
DataPack Creation Controller
===========================
CLI interface để tạo MatchZoo DataPacks từ chunked TSV files.

Usage Examples:
    # Tạo datapack từ chunked TSV
    python datapack_controller.py --input training_datasets/semantic_grouping_balanced_chunks.tsv --output ./datapacks/semantic_grouping

    # Với train/test split custom
    python datapack_controller.py --input training_datasets/semantic_grouping_balanced_chunks.tsv --output ./datapacks/semantic_grouping --train-ratio 0.8

    # Với cross-validation folds
    python datapack_controller.py --input training_datasets/semantic_grouping_balanced_chunks.tsv --output ./datapacks/semantic_grouping --cv-folds 5

    # Process tất cả chunked files trong thư mục
    python datapack_controller.py --input-dir training_datasets --output-dir ./datapacks --pattern "*_chunks.tsv"
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional
import glob

# Import từ MatchZoo_Tool
sys.path.insert(0, str(Path(__file__).parent / "MatchZoo_Tool"))
try:
    from create_matchzoo_datapacks import create_and_split_datapack, create_cv_folds_datapack
except ImportError:
    print("Không tìm thấy create_matchzoo_datapacks module")
    print("   Hãy đảm bảo file MatchZoo_Tool/create_matchzoo_datapacks.py tồn tại")
    sys.exit(1)

# ===== Utility Functions =====
def validate_tsv_file(file_path: Path) -> bool:
    """
    Validate TSV file có format đúng cho MatchZoo.
    Expected columns: query_id, document_id, chunk_text, label
    """
    try:
        df = pd.read_csv(file_path, sep='\t', nrows=5)
        expected_cols = ['query_id', 'document_id', 'chunk_text', 'label']
        
        if not all(col in df.columns for col in expected_cols):
            print(f"File {file_path} không có đủ columns cần thiết.")
            print(f"   Có: {list(df.columns)}")
            print(f"   Cần: {expected_cols}")
            return False
        
        return True
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return False

def convert_to_matchzoo_format(input_file: Path, output_file: Path) -> bool:
    """
    Convert chunked TSV format thành MatchZoo 3-column format.
    Input: query_id, document_id, chunk_text, label
    Output: query_text, chunk_text, label
    """
    try:
        print(f"🔄 Converting {input_file.name} to MatchZoo format...")
        
        df = pd.read_csv(input_file, sep='\t', dtype=str)
        
        # Tạo query_text từ query_id (có thể cần mapping nếu có)
        # Tạm thời dùng query_id làm query_text
        df_matchzoo = pd.DataFrame({
            'query': df['query_id'],  # Hoặc có thể map to actual query text
            'passage': df['chunk_text'],
            'label': df['label'].astype(int)
        })
        
        # Remove duplicates và invalid entries
        df_matchzoo = df_matchzoo.dropna()
        df_matchzoo = df_matchzoo[df_matchzoo['passage'].str.strip() != '']
        
        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_matchzoo.to_csv(output_file, sep='\t', index=False)
        
        print(f"   Converted: {len(df):,} → {len(df_matchzoo):,} samples")
        return True
        
    except Exception as e:
        print(f"Lỗi conversion: {e}")
        return False

def create_datapack_from_chunked_tsv(input_file: Path, 
                                   output_dir: Path,
                                   train_ratio: float = 0.8,
                                   create_cv_folds: bool = True,
                                   n_folds: int = 5) -> bool:
    """
    Tạo MatchZoo DataPack từ chunked TSV file.
    """
    try:
        # Validate input
        if not validate_tsv_file(input_file):
            return False
        
        # Convert to MatchZoo format
        temp_file = output_dir / f"{input_file.stem}_matchzoo_format.tsv"
        if not convert_to_matchzoo_format(input_file, temp_file):
            return False
        
        print(f" Creating DataPack từ {temp_file}")
        print(f"   Output: {output_dir}")
        print(f"   Train ratio: {train_ratio}")
        print(f"   CV folds: {n_folds if create_cv_folds else 'None'}")
        
        # Create DataPack
        create_and_split_datapack(
            input_file=str(temp_file),
            train_ratio=train_ratio,
            has_header=True,
            create_cv_folds=create_cv_folds,
            n_folds=n_folds
        )
        
        # Cleanup temp file
        temp_file.unlink()
        
        print(f"   DataPack tạo thành công!")
        return True
        
    except Exception as e:
        print(f" Lỗi tạo DataPack: {e}")
        return False

# ===== Batch Processing =====
def process_chunked_directory(input_dir: Path,
                             output_dir: Path,
                             pattern: str = "*_chunks.tsv",
                             train_ratio: float = 0.8,
                             create_cv_folds: bool = True,
                             n_folds: int = 5) -> List[str]:
    """
    Process tất cả chunked TSV files trong một directory.
    """
    chunked_files = list(input_dir.glob(pattern))
    
    if not chunked_files:
        print(f"Không tìm thấy files với pattern '{pattern}' trong {input_dir}")
        return []
    
    print(f"Tìm thấy {len(chunked_files)} chunked files:")
    for f in chunked_files:
        print(f"   - {f.name}")
    
    processed = []
    
    for chunked_file in chunked_files:
        method_name = chunked_file.stem.replace('_chunks', '')
        method_output_dir = output_dir / method_name
        
        print(f"\n{'='*50}")
        print(f"Processing: {method_name}")
        print(f"{'='*50}")
        
        if create_datapack_from_chunked_tsv(
            input_file=chunked_file,
            output_dir=method_output_dir,
            train_ratio=train_ratio,
            create_cv_folds=create_cv_folds,
            n_folds=n_folds
        ):
            processed.append(method_name)
            print(f"{method_name} completed")
        else:
            print(f"{method_name} failed")
    
    return processed

# ===== CLI Functions =====
def parse_arguments():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description="DataPack Creation Controller cho MatchZoo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python datapack_controller.py --input semantic_grouping_chunks.tsv --output ./datapacks/semantic_grouping
  
  # Batch processing
  python datapack_controller.py --input-dir training_datasets --output-dir ./datapacks
  
  # Custom parameters
  python datapack_controller.py --input file.tsv --output ./output --train-ratio 0.85 --cv-folds 10
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=Path,
        help='Input chunked TSV file'
    )
    input_group.add_argument(
        '--input-dir',
        type=Path,
        help='Directory chứa chunked TSV files'
    )
    
    # Output options
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory cho single file processing'
    )
    output_group.add_argument(
        '--output-dir',
        type=Path,
        help='Base output directory cho batch processing'
    )
    
    parser.add_argument(
        '--pattern',
        default='*_chunks.tsv',
        help='File pattern cho batch processing (default: *_chunks.tsv)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Tỉ lệ train/test split (default: 0.8)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Số lượng CV folds (default: 5, set 0 để tắt)'
    )
    
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Không tạo cross-validation folds'
    )
    
    return parser.parse_args()

def main():
    """Main CLI entry point"""
    args = parse_arguments()
    
    # Validate inputs
    if args.input and not args.input.exists():
        print(f"Input file không tồn tại: {args.input}")
        sys.exit(1)
    
    if args.input_dir and not args.input_dir.exists():
        print(f"Input directory không tồn tại: {args.input_dir}")
        sys.exit(1)
    
    # Validate train ratio
    if not 0.1 <= args.train_ratio <= 0.9:
        print("Train ratio phải trong khoảng 0.1 - 0.9")
        sys.exit(1)
    
    # CV folds settings
    create_cv = not args.no_cv and args.cv_folds > 0
    n_folds = max(2, args.cv_folds) if create_cv else 5
    
    try:
        if args.input and args.output:
            # Single file processing
            print(f"Single file mode")
            print(f"   Input: {args.input}")
            print(f"   Output: {args.output}")
            
            success = create_datapack_from_chunked_tsv(
                input_file=args.input,
                output_dir=args.output,
                train_ratio=args.train_ratio,
                create_cv_folds=create_cv,
                n_folds=n_folds
            )
            
            if success:
                print(f"\nDataPack tạo thành công tại: {args.output}")
                print(f"Sử dụng train_controller.py để train models!")
            else:
                print("\nTạo DataPack thất bại")
                sys.exit(1)
        
        elif args.input_dir and args.output_dir:
            # Batch processing
            print(f"Batch processing mode")
            print(f"   Input dir: {args.input_dir}")
            print(f"   Output dir: {args.output_dir}")
            print(f"   Pattern: {args.pattern}")
            
            processed = process_chunked_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                train_ratio=args.train_ratio,
                create_cv_folds=create_cv,
                n_folds=n_folds
            )
            
            print(f"\n{'='*50}")
            print(f"BATCH PROCESSING HOÀN THÀNH")
            print(f"{'='*50}")
            print(f"Processed: {len(processed)} methods")
            for method in processed:
                print(f"   - {method}")
            print(f"\n💡 Sử dụng train_controller.py để train models!")
        
        else:
            print("Invalid argument combination")
            sys.exit(1)
            
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
