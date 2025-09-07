import pandas as pd
import matchzoo as mz
import numpy as np
import os
import sys
import argparse
from typing import cast, Optional
from sklearn.model_selection import KFold

# ===== Helpers for normalization =====
def _normalize_text_series(s):
    try:
        s = s.astype(str)
        s = s.str.replace('\t', ' ', regex=False)
        s = s.str.replace('\r', ' ', regex=False)
        s = s.str.replace('\n', ' ', regex=False)
        s = s.str.strip()
        # Nén khoảng trắng thừa
        s = s.str.replace(r"\s+", " ", regex=True)
        return s
    except Exception:
        return s

def _normalize_label_value(v):
    """Map label theo các giá trị phổ biến, sau đó ép về int nếu là số."""
    if v is None:
        return None
    try:
        s = str(v).strip().lower()
        if s == "" or s == "nan":
            return None
        # Map boolean/text to numeric
        pos_set = {"1", "+1", "true", "yes", "y", "relevant", "rel", "pos", "+", "positive"}
        neg_set = {"0", "-1", "false", "no", "n", "irrelevant", "nonrelevant", "neg", "-", "negative"}
        if s in pos_set:
            return 1
        if s in neg_set:
            # Với IR có thể coi -1 là 0 (không liên quan)
            return 0
        # Thử parse số
        fv = float(s)
        if fv != fv:
            return None
        # Giữ nguyên miền giá trị nhưng cast int cho MatchZoo
        return int(fv)
    except Exception:
        return None

# ===== Hardcoded defaults (optional one-shot mode) =====
# Adjust these paths to your environment, then run:
#   python create_matchzoo_datapacks.py --use-hardcode
HARD_CODED_CONFIG = {
    # Input should be a 3-column TSV: query_text (or query), chunk_text (or passage), label
    # Recommended: the mapping output produced by controller ranking: *_with_querytext.tsv
    "input_file": "F:/SematicSearch/final_with_querytext.tsv",
    # Where to write cv_folds/*.dam files
    "output_dir": "datapacks/semantic_splitter_global",
    # Data properties
    "has_header": True,
    # CV controls
    "create_cv_folds": True,
    "n_folds": 5,
    # Unused when create_cv_folds is True (kept for completeness)
    "train_ratio": 0.8,
    # Verbose logging
    "verbose": True,
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Tạo MatchZoo DataPacks cho Cross-Validation từ file TSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Ví dụ sử dụng:
        python create_matchzoo_datapacks.py input.tsv
        python create_matchzoo_datapacks.py input.tsv --no-header
        python create_matchzoo_datapacks.py input.tsv --n-folds 10
        python create_matchzoo_datapacks.py input.tsv --output-dir ./datapacks
        python create_matchzoo_datapacks.py input.tsv --train-ratio 0.7 --no-cv
        
    Chạy với cấu hình hardcode (không cần đối số):
    python create_matchzoo_datapacks.py --use-hardcode
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Đường dẫn đến file TSV input (query_id | document_id | chunk_text | label)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Thư mục output (mặc định: cùng thư mục với input file)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Tỷ lệ train (0-1) - không sử dụng khi tạo CV folds (mặc định: 0.8)'
    )
    
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='File input không có header'
    )
    
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Không tạo Cross-Validation folds'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Số lượng folds cho Cross-Validation (mặc định: 5)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='In thêm thông tin chi tiết'
    )
    
    return parser.parse_args()

def interactive_mode():
    """Interactive mode như cũ"""
    print("=== INTERACTIVE MODE ===")
    
    while True:
        input_file = input("Nhập đường dẫn đến file input TSV: ")
        if os.path.isfile(input_file) and input_file.endswith('.tsv'):
            break
        else:
            print("File không tồn tại hoặc không phải file .tsv. Vui lòng nhập lại.")

    while True:
        try:
            train_ratio_str = input("Nhập tỉ lệ train (mặc định 0.8, phần còn lại sẽ là test): ")
            if not train_ratio_str:
                train_ratio = 0.8
                break
            train_ratio = float(train_ratio_str)
            if 0 < train_ratio < 1:
                break
            else:
                print("Tỉ lệ train phải là số trong khoảng (0, 1).")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ.")

    test_ratio = 1.0 - train_ratio
    print(f"Tỉ lệ test được tính tự động: {test_ratio:.2f}")
    print("Note: Train ratio sẽ không được sử dụng khi chỉ tạo CV folds.\n")
    
    # Hỏi về Cross-Validation (mặc định là Yes)
    while True:
        cv_choice = input("Tạo 5-fold cross-validation? (y/n, mặc định y): ").strip().lower()
        if cv_choice in ['yes', 'y', '']:
            create_cv_folds = True
            break
        elif cv_choice in ['no', 'n']:
            create_cv_folds = False
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")

    if create_cv_folds:
        print("Sẽ tạo 5-fold cross-validation từ toàn bộ dataset")
    else:
        print("Không tạo cross-validation folds. Script sẽ kết thúc.")

    while True:
        has_header_str = input("File có header không? (yes/no, mặc định yes): ").lower()
        if has_header_str in ['yes', 'y', '']:
            has_header = True
            break
        elif has_header_str in ['no', 'n']:
            has_header = False
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")

    create_and_split_datapack(input_file, train_ratio, has_header, create_cv_folds)
    
    if not create_cv_folds:
        print("\nKhông có dữ liệu nào được tạo vì Cross-Validation bị tắt.")
        print("Script này hiện chỉ hỗ trợ tạo CV folds.")
    else:
        print("\nHoàn thành! Các CV folds đã được tạo.")

def main():
    """Main function với CLI support"""
    # Nếu không có arguments, chạy interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return
    
    # One-shot hardcoded mode: allow running with just --use-hardcode
    if len(sys.argv) == 2 and sys.argv[1] in ("--use-hardcode", "--use-hardcoded"):
        cfg = HARD_CODED_CONFIG
        input_file = cfg["input_file"]
        output_dir = cfg["output_dir"] or os.path.dirname(input_file)
        os.makedirs(output_dir, exist_ok=True)

        print("=== CONFIGURATION (HARDCODE) ===")
        print(f"Input file    : {input_file}")
        print(f"Output dir    : {output_dir}")
        print(f"Has header    : {cfg['has_header']}")
        print(f"Create CV     : {cfg['create_cv_folds']}")
        if cfg['create_cv_folds']:
            print(f"Number of folds: {cfg['n_folds']}")
        print(f"Verbose       : {cfg['verbose']}")
        print("=" * 22)

        try:
            create_and_split_datapack(
                input_file=input_file,
                train_ratio=cfg.get('train_ratio', 0.8),
                has_header=cfg.get('has_header', True),
                create_cv_folds=cfg.get('create_cv_folds', True),
                n_folds=cfg.get('n_folds', 5),
                output_dir=output_dir,
            )
            print(f"\nHoàn thành! {cfg.get('n_folds', 5)}-fold cross-validation đã được tạo trong {os.path.join(output_dir, 'cv_folds')}/")
        except Exception as e:
            print(f"Lỗi (hardcode): {e}")
            sys.exit(1)
        return
    
    # Parse CLI arguments
    args = parse_arguments()
    
    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"Lỗi: File '{args.input_file}' không tồn tại.")
        sys.exit(1)
    
    if not args.input_file.endswith('.tsv'):
        print(f"Lỗi: File '{args.input_file}' không phải file .tsv.")
        sys.exit(1)
    
    # Validate train ratio
    if not (0 < args.train_ratio < 1):
        print(f"Lỗi: Train ratio phải trong khoảng (0, 1). Nhận được: {args.train_ratio}")
        sys.exit(1)
    
    # Validate n_folds
    if args.n_folds < 2:
        print(f"Lỗi: Số folds phải >= 2. Nhận được: {args.n_folds}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(args.input_file)
    
    # Print configuration
    print("=== CONFIGURATION ===")
    print(f"Input file    : {args.input_file}")
    print(f"Output dir    : {output_dir}")
    print(f"Has header    : {not args.no_header}")
    print(f"Train ratio   : {args.train_ratio} (unused in CV mode)")
    print(f"Create CV     : {not args.no_cv}")
    if not args.no_cv:
        print(f"Number of folds: {args.n_folds}")
    print(f"Verbose       : {args.verbose}")
    print("=" * 22)
    
    # Execute main logic
    try:
        create_and_split_datapack(
            input_file=args.input_file,
            train_ratio=args.train_ratio,
            has_header=not args.no_header,
            create_cv_folds=not args.no_cv,
            n_folds=args.n_folds if not args.no_cv else 5,
            output_dir=output_dir,
        )
        
        if args.no_cv:
            print("\nKhông có dữ liệu nào được tạo vì Cross-Validation bị tắt.")
            print("Script này hiện chỉ hỗ trợ tạo CV folds.")
        else:
            print(f"\nHoàn thành! {args.n_folds}-fold cross-validation đã được tạo trong {output_dir}/cv_folds/")
            
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

def create_and_split_datapack(input_file: str, train_ratio: float, has_header: bool, create_cv_folds: bool = True, n_folds: int = 5, output_dir: Optional[str] = None):
    """
    Đọc file TSV, chuyển đổi thành các DataPack của MatchZoo cho Cross-Validation.
    
    :param input_file: Đường dẫn đến file TSV.
    :param train_ratio: Tỷ lệ cho tập huấn luyện (không sử dụng khi chỉ tạo CV folds).
    :param has_header: True nếu file có dòng tiêu đề.
    :param create_cv_folds: True để tạo cross-validation (mặc định True).
    :param n_folds: Số lượng folds cho CV
    
    Note: Script này được sửa đổi để chỉ tạo Cross-Validation folds, 
          không tạo train_pack.dam và test_pack.dam riêng biệt.
    """
    print(f"Đang xử lý file: {input_file}")
    # Determine output directory
    output_dir = output_dir or os.path.dirname(input_file)
    
    # Check file size first to determine processing strategy
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    # For very large files (>500MB), use chunked processing
    if file_size_mb > 500:
        print("Large file detected. Using memory-optimized processing...")
        create_cv_folds_from_large_file(input_file, has_header, n_folds, output_dir)
        return

    # Original processing for smaller files
    # 1) Đọc file TSV
    try:
        if has_header:
            df_raw = cast(pd.DataFrame, pd.read_csv(input_file, sep='\t', header=0))
        else:
            df_raw = cast(pd.DataFrame, pd.read_csv(input_file, sep='\t', header=None))
            # Nếu không có header, gán tên cột mặc định
            if len(df_raw.columns) == 3:
                df_raw.columns = ['query', 'passage', 'label']
            else:
                print(f"Lỗi: File không có header và không có 3 cột. Số cột hiện tại: {len(df_raw.columns)}")
                sys.exit(1)

    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        sys.exit(1)

    # 2) Chuẩn hoá tên cột (tự động dò theo header nếu có)
    def _select_columns_by_name(df_in: pd.DataFrame):
        cols = [str(c).strip().lower() for c in df_in.columns]
        name_map = {c: df_in.columns[i] for i, c in enumerate(cols)}
        left_names = ['query_text', 'query', 'q', 'question']
        right_names = ['chunk_text', 'passage', 'document', 'doc', 'text', 'answer']
        label_names = ['label', 'relevance', 'rel', 'y']

        left_col = next((name_map[n] for n in left_names if n in name_map), None)
        right_col = next((name_map[n] for n in right_names if n in name_map), None)
        label_col = next((name_map[n] for n in label_names if n in name_map), None)
        return left_col, right_col, label_col

    left_col, right_col, label_col = _select_columns_by_name(df_raw) if has_header else (None, None, None)

    if left_col and right_col and label_col:
        use_df = df_raw[[left_col, right_col, label_col]].copy()
        use_df.columns = ['text_left', 'text_right', 'label']
    else:
        # Fallback: dùng 3 cột đầu
        if len(df_raw.columns) < 3:
            print(f"Lỗi: File cần tối thiểu 3 cột, nhận được {len(df_raw.columns)}")
            sys.exit(1)
        use_df = df_raw.iloc[:, :3].copy()
        use_df.columns = ['text_left', 'text_right', 'label']

    df = cast(pd.DataFrame, use_df)

    # 3) Làm sạch dữ liệu (chuẩn hoá text và label)
    df['text_left'] = _normalize_text_series(df['text_left'])
    df['text_right'] = _normalize_text_series(df['text_right'])
    df['label'] = df['label'].map(_normalize_label_value)
    df.dropna(subset=['text_left', 'text_right', 'label'], inplace=True)
    df = df[(df['text_left'] != '') & (df['text_right'] != '')]
    df['label'] = df['label'].astype(int)
    df.reset_index(drop=True, inplace=True)
    
    if len(df) == 0:
        print("Lỗi: Không có dữ liệu hợp lệ sau khi làm sạch.")
        sys.exit(1)
        
    print(f"Tìm thấy {len(df)} dòng dữ liệu hợp lệ.")
    # Sanity check: số lượng text_left duy nhất quá lớn có thể là sai cột (ví dụ dùng document_id thay vì query_text)
    try:
        uniq_left = int(df['text_left'].nunique())
        uniq_right = int(df['text_right'].nunique())
        print(f"Unique text_left: {uniq_left}, Unique text_right: {uniq_right}")
        if uniq_left > 5000:
            print("[Cảnh báo] Số lượng text_left duy nhất rất lớn. Có thể bạn đã chọn sai cột cho query_text.")
            print("[Gợi ý] Đảm bảo cột đầu tiên là query_text (không phải query_id/document_id).")
    except Exception:
        pass

    # 4) Đóng gói thành DataPack
    pack = mz.pack(df[['text_left', 'text_right', 'label']])
    print("Đã tạo DataPack thành công.")

    # 5) Xáo trộn dữ liệu
    np.random.seed(42)  # Để đảm bảo kết quả chia nhất quán
    pack = cast(mz.DataPack, pack.shuffle())
    
    print(f"Tổng số dữ liệu: {len(pack)} mẫu")

    # 6) Tạo Cross-Validation folds
    try:
        if create_cv_folds:
            print(f"\n--- Tạo {n_folds}-Fold Cross-Validation ---")
            create_cv_folds_datapack(pack, output_dir, n_folds)
        else:
            print("Không tạo cross-validation folds.")
            
    except Exception as e:
        print(f"Lỗi khi tạo CV folds: {e}")
        sys.exit(1)


def create_cv_folds_from_large_file(input_file: str, has_header: bool, n_folds: int, output_dir: str):
    """
    Xử lý file lớn bằng cách đọc từng chunk và tạo CV folds trực tiếp
    """
    try:
        import gc
        
        cv_dir = os.path.join(output_dir, 'cv_folds')
        os.makedirs(cv_dir, exist_ok=True)
        
        print("Processing large file with chunked reading...")
        
        # First pass: count total rows
        print("Counting total rows...")
        total_rows = 0
        chunksize = 10000
        
        for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunksize, header=0 if has_header else None):
            if not has_header and len(chunk.columns) == 3:
                chunk.columns = ['text_left', 'text_right', 'label']
            elif has_header:
                chunk = chunk.rename(columns={
                    chunk.columns[0]: 'text_left',
                    chunk.columns[1]: 'text_right',
                    chunk.columns[2]: 'label'
                })
            
            # Clean chunk: normalize text and label mapping
            chunk['text_left'] = _normalize_text_series(chunk['text_left'])
            chunk['text_right'] = _normalize_text_series(chunk['text_right'])
            chunk['label'] = chunk['label'].map(_normalize_label_value)
            chunk.dropna(subset=['text_left', 'text_right', 'label'], inplace=True)
            chunk = chunk[(chunk['text_left'] != '') & (chunk['text_right'] != '')]
            total_rows += len(chunk)
            
        print(f"Total valid rows: {total_rows:,}")
        
        # Calculate fold boundaries
        fold_size = total_rows // n_folds
        remainder = total_rows % n_folds
        
        fold_boundaries = []
        current_pos = 0
        for i in range(n_folds):
            fold_len = fold_size + (1 if i < remainder else 0)
            fold_boundaries.append((current_pos, current_pos + fold_len))
            current_pos += fold_len
        
        print(f"Fold sizes: {[end - start for start, end in fold_boundaries]}")
        
        # Create temporary files for each fold
        temp_files = {
            f'fold_{i+1}_train': [] for i in range(n_folds)
        }
        temp_files.update({
            f'fold_{i+1}_test': [] for i in range(n_folds)
        })
        
        # Second pass: distribute data to folds
        print("Distributing data to folds...")
        current_row = 0
        
        for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunksize, header=0 if has_header else None):
            if not has_header and len(chunk.columns) == 3:
                chunk.columns = ['text_left', 'text_right', 'label']
            elif has_header:
                chunk = chunk.rename(columns={
                    chunk.columns[0]: 'text_left',
                    chunk.columns[1]: 'text_right', 
                    chunk.columns[2]: 'label'
                })
            
            # Clean chunk: normalize text and label mapping
            chunk['text_left'] = _normalize_text_series(chunk['text_left'])
            chunk['text_right'] = _normalize_text_series(chunk['text_right'])
            chunk['label'] = chunk['label'].map(_normalize_label_value)
            chunk.dropna(subset=['text_left', 'text_right', 'label'], inplace=True)
            chunk = chunk[(chunk['text_left'] != '') & (chunk['text_right'] != '')]
            chunk.reset_index(drop=True, inplace=True)
            
            for idx, row in chunk.iterrows():
                # Determine which fold this row belongs to for test
                test_fold = None
                for fold_idx, (start, end) in enumerate(fold_boundaries):
                    if start <= current_row < end:
                        test_fold = fold_idx
                        break
                
                if test_fold is not None:
                    # Add to test set of this fold
                    temp_files[f'fold_{test_fold+1}_test'].append(row.to_dict())
                    
                    # Add to train sets of all other folds
                    for train_fold in range(n_folds):
                        if train_fold != test_fold:
                            temp_files[f'fold_{train_fold+1}_train'].append(row.to_dict())
                
                current_row += 1
            
            if current_row % 50000 == 0:
                print(f"  Processed {current_row:,}/{total_rows:,} rows...")
        
        # Create DataPacks from accumulated data
        print("Creating DataPacks...")
        fold_info = []
        
        for fold_idx in range(n_folds):
            try:
                train_data = temp_files[f'fold_{fold_idx+1}_train']
                test_data = temp_files[f'fold_{fold_idx+1}_test']
                
                if train_data and test_data:
                    train_df = pd.DataFrame(train_data)
                    test_df = pd.DataFrame(test_data)
                    
                    train_pack = mz.pack(train_df)
                    test_pack = mz.pack(test_df)
                    
                    train_path = os.path.join(cv_dir, f'fold_{fold_idx+1}_train.dam')
                    test_path = os.path.join(cv_dir, f'fold_{fold_idx+1}_test.dam')
                    
                    train_pack.save(train_path)
                    test_pack.save(test_path)
                    
                    fold_info.append({
                        'fold': fold_idx + 1,
                        'train_size': len(train_pack),
                        'test_size': len(test_pack),
                        'train_path': train_path,
                        'test_path': test_path
                    })
                    
                    print(f"✓ Fold {fold_idx+1}: {len(train_pack):,} train, {len(test_pack):,} test")
                    
                    del train_df, test_df, train_pack, test_pack
                    gc.collect()
                    
            except Exception as e:
                print(f"✗ Error creating fold {fold_idx+1}: {e}")
        
        # Save fold info
        if fold_info:
            fold_info_path = os.path.join(cv_dir, 'fold_info.txt')
            with open(fold_info_path, 'w', encoding='utf-8') as f:
                f.write(f"{n_folds}-Fold Cross-Validation Information (Large File Processing)\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total dataset size: {total_rows:,} samples\n\n")
                
                for info in fold_info:
                    f.write(f"Fold {info['fold']}:\n")
                    f.write(f"  Train: {info['train_size']:,} samples -> {info['train_path']}\n") 
                    f.write(f"  Test:  {info['test_size']:,} samples -> {info['test_path']}\n\n")
            
            print(f"\n✓ Successfully created {len(fold_info)}/{n_folds} folds using large file processing!")
            print(f"✓ Output directory: {cv_dir}")
            print(f"✓ Details saved to: {fold_info_path}")
        
    except Exception as e:
        print(f"✗ Error in large file processing: {e}")
        import traceback
        traceback.print_exc()

def create_cv_folds_datapack(full_pack: mz.DataPack, output_dir: str, n_folds: int = 5):
    """
    Tạo n-fold cross-validation từ toàn bộ dataset với memory optimization.
    
    :param full_pack: DataPack chứa toàn bộ dữ liệu để chia thành các folds
    :param output_dir: Thư mục lưu các folds
    :param n_folds: Số lượng folds (mặc định 5)
    """
    try:
        import gc
        from typing import List, Tuple
        
        # Tạo thư mục con cho CV folds
        cv_dir = os.path.join(output_dir, 'cv_folds')
        os.makedirs(cv_dir, exist_ok=True)
        
        print(f"Starting memory-optimized {n_folds}-fold CV creation...")
        
        # Lấy size của dataset
        total_size = len(full_pack)
        print(f"Total dataset size: {total_size:,} samples")
        
        # Tính fold size
        fold_size = total_size // n_folds
        remainder = total_size % n_folds
        
        print(f"Base fold size: {fold_size:,} samples")
        if remainder > 0:
            print(f"Some folds will have +1 sample (remainder: {remainder})")
        
        # Tạo indices cho từng fold
        fold_indices = []
        start_idx = 0
        
        for fold_idx in range(n_folds):
            # Some folds get +1 sample for remainder
            current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
            end_idx = start_idx + current_fold_size
            
            fold_indices.append((start_idx, end_idx))
            start_idx = end_idx
        
        fold_info = []
        
        # Xử lý từng fold một để tránh memory overflow
        for fold_idx in range(n_folds):
            print(f"\nCreating Fold {fold_idx + 1}/{n_folds}...")
            
            # Lấy test indices cho fold hiện tại
            test_start, test_end = fold_indices[fold_idx]
            test_size = test_end - test_start
            
            # Train indices là tất cả indices khác
            train_size = total_size - test_size
            
            print(f"  Train size: {train_size:,}, Test size: {test_size:,}")
            
            try:
                # Tạo train pack (tất cả trừ fold hiện tại)
                train_data_left = []
                train_data_right = []
                train_labels = []
                
                # Collect train data from other folds in chunks to manage memory
                chunk_size = min(10000, fold_size)  # Process in smaller chunks
                
                for other_fold_idx in range(n_folds):
                    if other_fold_idx == fold_idx:
                        continue  # Skip current test fold
                        
                    other_start, other_end = fold_indices[other_fold_idx]
                    
                    # Process this fold in chunks
                    for chunk_start in range(other_start, other_end, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, other_end)
                        
                        # Extract chunk from full_pack
                        chunk_indices = list(range(chunk_start, chunk_end))
                        chunk_pack = full_pack[chunk_indices]
                        
                        # Unpack chunk data
                        chunk_data, chunk_labels = chunk_pack.unpack()
                        
                        # Append to train data
                        train_data_left.extend(chunk_data['text_left'])
                        train_data_right.extend(chunk_data['text_right'])
                        train_labels.extend(chunk_labels.flatten())
                        
                        # Clean up chunk to free memory
                        del chunk_pack, chunk_data, chunk_labels
                        gc.collect()
                
                # Create train DataFrame and pack
                train_df = pd.DataFrame({
                    'text_left': train_data_left,
                    'text_right': train_data_right,
                    'label': train_labels
                })
                
                fold_train_pack = mz.pack(train_df)
                
                # Clean up train data to free memory
                del train_data_left, train_data_right, train_labels, train_df
                gc.collect()
                
                # Create test pack
                test_indices = list(range(test_start, test_end))
                fold_test_pack = full_pack[test_indices]
                
                # Save fold packs
                fold_train_path = os.path.join(cv_dir, f'fold_{fold_idx + 1}_train.dam')
                fold_test_path = os.path.join(cv_dir, f'fold_{fold_idx + 1}_test.dam')
                
                fold_train_pack.save(fold_train_path)
                fold_test_pack.save(fold_test_path)
                
                fold_info.append({
                    'fold': fold_idx + 1,
                    'train_size': len(fold_train_pack),
                    'test_size': len(fold_test_pack),
                    'train_path': fold_train_path,
                    'test_path': fold_test_path
                })
                
                print(f"  ✓ Saved: {len(fold_train_pack):,} train, {len(fold_test_pack):,} test")
                
                # Clean up fold packs
                del fold_train_pack, fold_test_pack
                gc.collect()
                
            except Exception as e:
                print(f"  ✗ Error creating fold {fold_idx + 1}: {e}")
                continue
        
        # Lưu thông tin về các folds
        if fold_info:
            fold_info_path = os.path.join(cv_dir, 'fold_info.txt')
            with open(fold_info_path, 'w', encoding='utf-8') as f:
                f.write(f"{n_folds}-Fold Cross-Validation Information\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total dataset size: {total_size:,} samples\n\n")
                
                total_train = 0
                total_test = 0
                
                for info in fold_info:
                    f.write(f"Fold {info['fold']}:\n")
                    f.write(f"  Train: {info['train_size']:,} samples -> {info['train_path']}\n")
                    f.write(f"  Test:  {info['test_size']:,} samples -> {info['test_path']}\n\n")
                    
                    total_train += info['train_size']
                    total_test += info['test_size']
                
                f.write(f"Summary:\n")
                f.write(f"  Successfully created: {len(fold_info)}/{n_folds} folds\n")
                f.write(f"  Average train size: {total_train // len(fold_info):,} samples\n")
                f.write(f"  Average test size: {total_test // len(fold_info):,} samples\n")

            print(f"\n✓ Successfully created {len(fold_info)}/{n_folds} folds!")
            print(f"✓ Output directory: {cv_dir}")
            print(f"✓ Details saved to: {fold_info_path}")
        else:
            print(f"\n✗ Failed to create any folds!")
        
    except Exception as e:
        print(f"✗ Error in CV folds creation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()