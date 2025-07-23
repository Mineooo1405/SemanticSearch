import pandas as pd
import matchzoo as mz
import numpy as np
import os
import sys
from typing import cast
from sklearn.model_selection import KFold

def create_and_split_datapack(input_file: str, train_ratio: float, has_header: bool, create_cv_folds: bool = False):
    """
    Đọc file TSV, chuyển đổi thành các DataPack của MatchZoo (train và test), và lưu chúng lại.
    
    :param input_file: Đường dẫn đến file TSV.
    :param train_ratio: Tỷ lệ cho tập huấn luyện (phần còn lại sẽ là test).
    :param has_header: True nếu file có dòng tiêu đề.
    :param create_cv_folds: True nếu muốn tạo 5-fold cross-validation.
    
    Note: MatchZoo thường chỉ sử dụng train_pack và test_pack. 
          Validation được thực hiện thông qua validation_split trong quá trình training.
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

    # 5) Xáo trộn và phân chia thành train/test
    np.random.seed(42)  # Để đảm bảo kết quả chia nhất quán
    pack = cast(mz.DataPack, pack.shuffle())
    
    total_size = len(pack)
    train_end = int(total_size * train_ratio)

    train_pack = pack[:train_end]
    test_pack = pack[train_end:]

    print(f"Đã chia dữ liệu: {len(train_pack)} train, {len(test_pack)} test.")
    print("Note: Validation sẽ được thực hiện thông qua validation_split trong quá trình training.")

    # 6) Lưu các DataPack
    try:
        train_pack.save(os.path.join(output_dir, 'train_pack.dam'))
        test_pack.save(os.path.join(output_dir, 'test_pack.dam'))
        print(f"Đã lưu train_pack.dam và test_pack.dam vào thư mục: {output_dir}")
        
        # 7) Tạo Cross-Validation folds nếu được yêu cầu
        if create_cv_folds:
            print("\n--- Tạo 5-Fold Cross-Validation ---")
            create_cv_folds_datapack(train_pack, output_dir)
            
    except Exception as e:
        print(f"Lỗi khi lưu file DataPack: {e}")
        sys.exit(1)

def create_cv_folds_datapack(train_pack: mz.DataPack, output_dir: str, n_folds: int = 5):
    """
    Tạo 5-fold cross-validation từ train_pack.
    
    :param train_pack: DataPack để chia thành các folds
    :param output_dir: Thư mục lưu các folds
    :param n_folds: Số lượng folds (mặc định 5)
    """
    try:
        # Tạo thư mục con cho CV folds
        cv_dir = os.path.join(output_dir, 'cv_folds')
        os.makedirs(cv_dir, exist_ok=True)
        
        # Lấy dữ liệu từ train_pack để xử lý với KFold
        # MatchZoo DataPack.unpack() trả về tuple (data_dict, labels_array)
        data_dict, labels_array = train_pack.unpack()
        
        # Tạo DataFrame từ dữ liệu đã unpack
        df_train = pd.DataFrame({
            'text_left': data_dict['text_left'],
            'text_right': data_dict['text_right'], 
            'label': labels_array.flatten()  # flatten để chuyển từ 2D array thành 1D
        })
        
        print(f"Dữ liệu CV: {len(df_train)} mẫu")
        
        # Tạo KFold splitter
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_info = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(df_train)):
            print(f"Đang tạo Fold {fold_idx + 1}/{n_folds}...")
            
            # Chia dữ liệu theo indices
            fold_train_df = df_train.iloc[train_indices].reset_index(drop=True)
            fold_test_df = df_train.iloc[test_indices].reset_index(drop=True)
            
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

        
    except Exception as e:
        print(f"Lỗi khi tạo CV folds: {e}")
        return

if __name__ == "__main__":
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
    # Hỏi về Cross-Validation
    while True:
        cv_choice = input("\nTạo 5-fold cross-validation? (y/n, mặc định n): ").strip().lower()
        if cv_choice in ['yes', 'y']:
            create_cv_folds = True
            break
        elif cv_choice in ['no', 'n', '']:
            create_cv_folds = False
            break
        else:
            print("Vui lòng nhập 'yes' hoặc 'no'.")

    if create_cv_folds:
        print("Sẽ tạo 5-fold cross-validation từ train data")
    else:
        print("Chỉ tạo train/test split thông thường")

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

