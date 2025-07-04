import sys
import csv

csv.field_size_limit(min(sys.maxsize, 2147483647))

def head_csv(filename, n=10):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for i, row in enumerate(reader):
            print('|'.join(row))
            if i + 1 >= n:
                break

def percent_text_tag(filename):
    total = 0
    with_text = 0
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        header = next(reader, None)  # skip header
        for row in reader:
            total += 1
            if len(row) > 0 and '[Text]' in row[-1]:  # Only check Data column
                with_text += 1
    if total == 0:
        print('No data rows found.')
        return 0.0
    percent = (with_text / total) * 100
    print(f"Rows with '[Text]' in Data column: {with_text}/{total} ({percent:.2f}%)")
    return percent

def show_first_n_no_text(filename, n=10):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        header = next(reader, None)
        print('|'.join(header))
        count = 0
        for row in reader:
            if len(row) > 0 and '[Text]' not in row[-1]:
                print('|'.join(row))
                count += 1
                if count >= n:
                    break

def percent_without_brackets(filename):
    total = 0
    without_brackets = 0
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        header = next(reader, None)  # skip header
        for row in reader:
            total += 1
            if len(row) > 0 and '[' not in row[-1] and ']' not in row[-1]:  # Check if data column doesn't contain any brackets
                without_brackets += 1
    if total == 0:
        print('No data rows found.')
        return 0.0
    percent = (without_brackets / total) * 100
    print(f"Rows without any '[]' in Data column: {without_brackets}/{total} ({percent:.2f}%)")
    print(f"Rows with '[]' in Data column: {total - without_brackets}/{total} ({100 - percent:.2f}%)")
    return percent

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--percent-text':
        filename = sys.argv[2] if len(sys.argv) > 2 else input('Enter CSV file path: ')
        percent_text_tag(filename)
    else:
        if len(sys.argv) < 2:
            filename = input("Enter CSV file path: ")
            mode = input("Type 'p' to print percentage of rows with '[Text]' in Data column, \n"
                        "'n' to show first 10 rows without '[Text]', \n"
                        "'b' to calculate percentage of rows without any square brackets, \n"
                        "or press Enter to display rows: ").strip().lower()
            if mode == 'p':
                percent_text_tag(filename)
            elif mode == 'n':
                show_first_n_no_text(filename, 10)
            elif mode == 'b':
                percent_without_brackets(filename)
            else:
                n = input("How many rows to display? (default 10): ")
                n = int(n) if n.strip() else 10
                head_csv(filename, n)
        else:
            filename = sys.argv[1]
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            head_csv(filename, n)