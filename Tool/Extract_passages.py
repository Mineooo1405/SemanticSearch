import pandas as pd
import ast

def extract_passage_texts(csv_file, limit=1000):
    
    df = pd.read_csv(csv_file)

    passages_with_id = []
    count = 0

    for idx, row in df.iterrows():
        try:
  
            passages_dict = ast.literal_eval(row['passages'])
 
            passage_texts = passages_dict['passage_text']
            is_selected = passages_dict['is_selected']

            selected_idx = is_selected.index(1) if 1 in is_selected else 0
            passage = passage_texts[selected_idx]

            count += 1
            passages_with_id.append({
                'id': count,
                'passage_text': passage
            })

            if count >= limit:
                break
                
        except Exception as e:
            print(f"Lỗi khi xử lý hàng {idx}: {e}")
    
    return passages_with_id

def save_to_csv(passages, output_file):

    df = pd.DataFrame(passages)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Đã lưu {len(passages)} đoạn văn vào {output_file}")

if __name__ == "__main__":
    input_file = "d:/SemanticSearch/ms_marco_1000.csv"

    passages = extract_passage_texts(input_file, limit=10)

    print(f"\nĐã trích xuất {len(passages)} đoạn văn")
    for i in range(min(3, len(passages))):
        print(f"\nĐoạn {i+1}:")
        print(f"ID: {passages[i]['id']}")
        print(f"Passage: {passages[i]['passage_text'][:150]}...")
        
    output_file = f"d:/SemanticSearch/passages_{len(passages)}.csv"
    save_to_csv(passages, output_file)