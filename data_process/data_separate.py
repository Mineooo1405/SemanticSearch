import sys
import random
import os

def split_tsv(input_file, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, has_header=True):
    # Đọc dữ liệu
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if has_header:
        header = lines[0]
        data = lines[1:]
    else:
        header = None
        data = lines

    # Xáo trộn dữ liệu
    random.shuffle(data)

    # Tính số lượng mẫu cho mỗi tập
    total = len(data)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_data = data[:train_end]
    dev_data = data[train_end:dev_end]
    test_data = data[dev_end:]

    # Ghi ra file
    def write_file(filename, data):
        with open(filename, 'w', encoding='utf-8') as f:
            if header:
                f.write(header)
            f.writelines(data)

    output_dir = os.path.dirname(input_file)

    write_file(os.path.join(output_dir, 'train.tsv'), train_data)
    write_file(os.path.join(output_dir, 'dev.tsv'), dev_data)
    write_file(os.path.join(output_dir, 'test.tsv'), test_data)
    print(f'Đã chia xong: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test')

if __name__ == "__main__":
    while True:
        input_file = input("Nhập đường dẫn đến file input: ")
        if os.path.isfile(input_file):
            break
        else:
            print("File không tồn tại. Vui lòng nhập lại.")

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

    test_ratio = 1.0 - train_ratio - dev_ratio
    print(f"Tỉ lệ test được tính tự động: {test_ratio:.2f}")
    if test_ratio <= 0:
        print("Lỗi: Tổng tỉ lệ train và dev phải nhỏ hơn 1.")
        sys.exit(1)

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

    split_tsv(input_file, train_ratio, dev_ratio, test_ratio, has_header)