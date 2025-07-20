import pandas as pd
import matchzoo as mz
import numpy as np
import os
import sys
from typing import cast

def create_and_split_datapack(input_file: str, train_ratio: float, dev_ratio: float, has_header: bool):
    """
    Đọc file TSV, chuyển đổi thành các DataPack của MatchZoo, và lưu chúng lại.
    
    :param input_file: Đường dẫn đến file TSV.
    :param train_ratio: Tỷ lệ cho tập huấn luyện.
    :param dev_ratio: Tỷ lệ cho tập phát triển.
    :param has_header: True nếu file có dòng tiêu đề.
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

    # 5) Xáo trộn và phân chia
    np.random.seed(42) # Để đảm bảo kết quả chia nhất quán
    pack = cast(mz.DataPack, pack.shuffle())
    
    total_size = len(pack)
    train_end = int(total_size * train_ratio)
    dev_end = train_end + int(total_size * dev_ratio)

    train_pack = pack[:train_end]
    dev_pack = pack[train_end:dev_end]
    test_pack = pack[dev_end:]

    print(f"Đã chia dữ liệu: {len(train_pack)} train, {len(dev_pack)} dev, {len(test_pack)} test.")

    # 6) Lưu các DataPack
    try:
        train_pack.save(os.path.join(output_dir, 'train_pack.dam'))
        dev_pack.save(os.path.join(output_dir, 'dev_pack.dam'))
        test_pack.save(os.path.join(output_dir, 'test_pack.dam'))
        print(f"Đã lưu các file DataPack vào thư mục: {output_dir}")
    except Exception as e:
        print(f"Lỗi khi lưu file DataPack: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("--- Công cụ tạo và chia DataPack cho MatchZoo ---")
    
    while True:
        input_file = input("Nhập đường dẫn đến file input TSV: ")
        if os.path.isfile(input_file) and input_file.endswith('.tsv'):
            break
        else:
            print("File không tồn tại hoặc không phải file .tsv. Vui lòng nhập lại.")

    while True:
        try:
            train_ratio_str = input("Nhập tỉ lệ train (mặc định 0.8): ")
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

    while True:
        try:
            dev_ratio_str = input("Nhập tỉ lệ dev (mặc định 0.1): ")
            if not dev_ratio_str:
                dev_ratio = 0.1
                break
            dev_ratio = float(dev_ratio_str)
            if 0 < dev_ratio < 1:
                break
            else:
                print("Tỉ lệ dev phải là số trong khoảng (0, 1).")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ.")

    if train_ratio + dev_ratio >= 1.0:
        print(f"Lỗi: Tổng tỉ lệ train ({train_ratio}) và dev ({dev_ratio}) phải nhỏ hơn 1.")
        sys.exit(1)
        
    test_ratio = 1.0 - train_ratio - dev_ratio
    print(f"Tỉ lệ test được tính tự động: {test_ratio:.2f}")

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

    create_and_split_datapack(input_file, train_ratio, dev_ratio, has_header)
    print("--- Hoàn thành! ---")
