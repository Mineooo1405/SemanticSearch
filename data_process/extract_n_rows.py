import csv
import os

def get_first_n_rows(file_path, n=10, include_header=True, delimiter=None):
    if delimiter is None:
        delimiter = detect_delimiter(file_path)
        print(f"Auto-detected delimiter: '{delimiter}'")
    
    try:
        rows = []
        with open(file_path, 'r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter=delimiter)
            
            # Đọc header
            header = next(reader, None)
            if header and include_header:
                rows.append(header)
                print(f"Header ({len(header)} columns): {header}")
            
            # Đọc n rows data
            for i, row in enumerate(reader):
                if i >= n:
                    break
                rows.append(row)
            
        print(f"Đã đọc {len(rows) - (1 if include_header else 0)} rows data từ file")
        return rows
        
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return []

def detect_delimiter(file_path, sample_lines=3):
    """Tự động phát hiện delimiter của file"""
    try:
        # Nếu file có extension .tsv, ưu tiên tab delimiter
        if file_path.lower().endswith('.tsv'):
            return '\t'
            
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = ""
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                sample += line
        
        # Đếm các delimiter phổ biến
        delimiters = {
            '\t': sample.count('\t'),
            ',': sample.count(','),
            '|': sample.count('|'),
            ';': sample.count(';')
        }
        
        return max(delimiters, key=delimiters.get)
        
    except Exception:
        return ','  # Default fallback

def generate_output_filename(input_path, n, suffix="_subset"):
    input_path = os.path.abspath(input_path)
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    # Tạo tên file mới
    output_filename = f"{name}{suffix}_{n}rows{ext}"
    output_path = os.path.join(directory, output_filename)
    
    return output_path

def save_rows_to_file(rows, output_path, delimiter='\t'):
    try:
        with open(output_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=delimiter)
            
            for row in rows:
                writer.writerow(row)
        
        return True
        
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")
        return False

def extract_and_save(input_path, n=5, include_header=True, output_path=None, delimiter=None):
    print(f"Bắt đầu extract {n} rows từ: {input_path}")
    
    # Lấy rows từ file input
    rows = get_first_n_rows(input_path, n, include_header, delimiter)

    if delimiter is None:
        delimiter = detect_delimiter(input_path)
    
    # Tạo output path nếu chưa có
    if output_path is None:
        output_path = generate_output_filename(input_path, n)
    
    print(f"lưu vào: {output_path}")
    
    # Lưu rows vào file
    if save_rows_to_file(rows, output_path, delimiter):
        print(f"Đã lưu thành công {len(rows)} rows vào: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
        return output_path
    else:
        return None

def print_rows_formatted(rows, max_col_width=50):
    if not rows:
        print("Không có dữ liệu để hiển thị")
        return
    
    print("\n" + "="*100)
    for i, row in enumerate(rows):
        row_type = "HEADER" if i == 0 else f"ROW {i}"
        print(f"{row_type}:")
        
        for j, cell in enumerate(row):
            cell_content = str(cell)
            if len(cell_content) > max_col_width:
                cell_content = cell_content[:max_col_width] + "..."
            
            print(f"  [{j}] {cell_content}")
        print("-" * 100)

def preview_file(file_path, n=5, include_header=True, delimiter=None, print_formatted=True):
    print(f"Đang preview file: {file_path}")
    print(f"Lấy {n} rows đầu tiên (include_header={include_header})")
    
    rows = get_first_n_rows(file_path, n, include_header, delimiter)
    
    if rows and print_formatted:
        print_rows_formatted(rows)
    
    return rows

if __name__ == "__main__":

    file_path = input("Nhập đường dẫn file: ").strip().replace('"', '')

    n = int(input("Số rows cần lấy (default=5): ") or "5")

    preview_choice = input("Xem preview trước khi lưu? (y/n, default=y): ").lower()
    if preview_choice != 'n':
        print("\nPREVIEW:")
        preview_file(file_path, n=min(n, 3))

    save_choice = input("\nLưu vào file mới? (y/n, default=y): ").lower()
    if save_choice != 'n':
        custom_output = input("Đường dẫn file output (để trống = auto-generate): ").strip().replace('"', '')
        output_path = custom_output if custom_output else None

        result_path = extract_and_save(file_path, n=n, output_path=output_path)
        
        if result_path:
            print(f"\nHOÀN THÀNH!")
            print(f"File gốc: {file_path}")
            print(f"File mới: {result_path}")
            
            # Hỏi có muốn xem preview file mới không
            preview_new = input("\nXem preview file mới? (y/n, default=n): ").lower()
            if preview_new == 'y':
                print("\nPREVIEW FILE MỚI:")
                preview_file(result_path, n=3)
        else:
            print("Lưu file thất bại")
    else:
        print("Chỉ preview, không lưu file")