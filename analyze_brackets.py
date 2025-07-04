import sys
import csv
import re
from collections import Counter

csv.field_size_limit(sys.maxsize)

def analyze_bracket_patterns(filename):
    """
    Analyze all bracket patterns in the Data column
    """
    bracket_patterns = Counter()
    rows_with_brackets = 0
    total_rows = 0
    
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        header = next(reader, None)  # Skip header
        
        for row in reader:
            total_rows += 1
            if len(row) > 0:
                data_text = row[-1]  # Last column (Data column)
                
                # Find all bracket patterns [word]
                matches = re.findall(r'\[([^\]]+)\]', data_text)
                
                if matches:
                    rows_with_brackets += 1
                    for match in matches:
                        bracket_patterns[match] += 1
    
    print(f"Total rows: {total_rows}")
    print(f"Rows with brackets: {rows_with_brackets}")
    print(f"Unique bracket patterns found: {len(bracket_patterns)}")
    print("\nTop 20 bracket patterns:")
    print("-" * 50)
    
    for pattern, count in bracket_patterns.most_common(20):
        percentage = (count / total_rows) * 100
        print(f"[{pattern}]: {count} times ({percentage:.2f}%)")
    
    return bracket_patterns

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = input("Enter CSV file path: ").strip()
    
    patterns = analyze_bracket_patterns(filename)