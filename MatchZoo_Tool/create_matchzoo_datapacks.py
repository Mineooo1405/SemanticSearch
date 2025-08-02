import pandas as pd
import matchzoo as mz
import numpy as np
import os
import sys
import argparse
from typing import cast
from sklearn.model_selection import KFold

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
            n_folds=args.n_folds if not args.no_cv else 5
        )
        
        if args.no_cv:
            print("\nKhông có dữ liệu nào được tạo vì Cross-Validation bị tắt.")
            print("Script này hiện chỉ hỗ trợ tạo CV folds.")
        else:
            print(f"\nHoàn thành! {args.n_folds}-fold cross-validation đã được tạo trong {output_dir}/cv_folds/")
            
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)

def create_and_split_datapack(input_file: str, train_ratio: float, has_header: bool, create_cv_folds: bool = True, n_folds: int = 5):
    """
    Đọc file TSV, chuyển đổi thành các DataPack của MatchZoo cho Cross-Validation.
    
    :param input_file: Đường dẫn đến file TSV.
    :param train_ratio: Tỷ lệ cho tập huấn luyện (không sử dụng khi chỉ tạo CV folds).
    :param has_header: True nếu file có dòng tiêu đề.
    :param create_cv_folds: True để tạo 5-fold cross-validation (mặc định True).
    
    Note: Script này được sửa đổi để chỉ tạo Cross-Validation folds, 
          không tạo train_pack.dam và test_pack.dam riêng biệt.
    """
    print(f"Đang xử lý file: {input_file}")
    output_dir = os.path.dirname(input_file)
    
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

    # 2) Chuẩn hoá tên cột
    df = cast(pd.DataFrame, df_raw.rename(columns={
        df_raw.columns[0]: 'text_left',
        df_raw.columns[1]: 'text_right',
        df_raw.columns[2]: 'label'
    }))

    # 3) Làm sạch dữ liệu
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df.dropna(subset=['text_left', 'text_right', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)
    df.reset_index(drop=True, inplace=True)
    
    if len(df) == 0:
        print("Lỗi: Không có dữ liệu hợp lệ sau khi làm sạch.")
        sys.exit(1)
        
    print(f"Tìm thấy {len(df)} dòng dữ liệu hợp lệ.")

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
            print("\n--- Tạo 5-Fold Cross-Validation ---")
            create_cv_folds_datapack(pack, output_dir, n_folds)
        else:
            print("Không tạo cross-validation folds.")
            
    except Exception as e:
        print(f"Lỗi khi tạo CV folds: {e}")
        sys.exit(1)

def create_cv_folds_datapack(full_pack: mz.DataPack, output_dir: str, n_folds: int = 5):
    """
    Tạo 5-fold cross-validation từ toàn bộ dataset.
    
    :param full_pack: DataPack chứa toàn bộ dữ liệu để chia thành các folds
    :param output_dir: Thư mục lưu các folds
    :param n_folds: Số lượng folds (mặc định 5)
    """
    try:
        # Tạo thư mục con cho CV folds
        cv_dir = os.path.join(output_dir, 'cv_folds')
        os.makedirs(cv_dir, exist_ok=True)
        
        # Lấy dữ liệu từ full_pack để xử lý với KFold
        # MatchZoo DataPack.unpack() trả về tuple (data_dict, labels_array)
        data_dict, labels_array = full_pack.unpack()
        
        # Tạo DataFrame từ dữ liệu đã unpack
        df_full = pd.DataFrame({
            'text_left': data_dict['text_left'],
            'text_right': data_dict['text_right'], 
            'label': labels_array.flatten()  # flatten để chuyển từ 2D array thành 1D
        })
        
        print(f"Dữ liệu CV: {len(df_full)} mẫu")
        
        # Tạo KFold splitter
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_info = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(df_full)):
            print(f"Đang tạo Fold {fold_idx + 1}/{n_folds}...")
            
            # Chia dữ liệu theo indices
            fold_train_df = df_full.iloc[train_indices].reset_index(drop=True)
            fold_test_df = df_full.iloc[test_indices].reset_index(drop=True)
            
            # Tạo DataPack cho từng fold
            fold_train_pack = mz.pack(fold_train_df)
            fold_test_pack = mz.pack(fold_test_df)
            
            # Lưu các fold
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
            
            print(f"  Fold {fold_idx + 1}: {len(fold_train_pack)} train, {len(fold_test_pack)} test")
        
        # Lưu thông tin về các folds
        fold_info_path = os.path.join(cv_dir, 'fold_info.txt')
        with open(fold_info_path, 'w', encoding='utf-8') as f:
            f.write("5-Fold Cross-Validation Information\n")
            f.write("=" * 40 + "\n\n")
            for info in fold_info:
                f.write(f"Fold {info['fold']}:\n")
                f.write(f"  Train: {info['train_size']} samples -> {info['train_path']}\n")
                f.write(f"  Test:  {info['test_size']} samples -> {info['test_path']}\n\n")

        print(f"\nĐã tạo {n_folds}-fold cross-validation thành công!")
        print(f"Thư mục output: {cv_dir}")
        print(f"Chi tiết folds: {fold_info_path}")
        
    except Exception as e:
        print(f"Lỗi khi tạo CV folds: {e}")
        return

if __name__ == "__main__":
    main()