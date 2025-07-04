import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
import json
import psycopg2
from Tool.Database import connect_to_db

load_dotenv()

dataset = load_dataset("microsoft/ms_marco", "v1.1")

passages = dataset['train'].select(range(1000))

# Convert to pandas DataFrame
df = pd.DataFrame(passages)

for col in df.columns:
    if isinstance(df[col].iloc[0], (dict, list)):
        df[col] = df[col].apply(str)

# Save to CSV file
csv_path = "ms_marco_1000.csv"
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"Data saved to {csv_path}")

def import_csv_to_database(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        conn = connect_to_db()
        
        cursor = conn.cursor()
        
        # Insert each row into the MS_MARCO table
        for index, row in df.iterrows():
            try:
                answers_json = json.dumps(eval(row['answers'])) if isinstance(row['answers'], str) else json.dumps(row['answers'])
                passages_json = json.dumps(eval(row['passages'])) if isinstance(row['passages'], str) else json.dumps(row['passages'])
                well_formed_json = json.dumps(eval(row['wellFormedAnswers'])) if isinstance(row['wellFormedAnswers'], str) else json.dumps(row['wellFormedAnswers'])
            except:
                answers_json = json.dumps(row['answers']) if row['answers'] else 'null'
                passages_json = json.dumps(row['passages']) if row['passages'] else 'null'
                well_formed_json = json.dumps(row['wellFormedAnswers']) if row['wellFormedAnswers'] else 'null'
            
            cursor.execute("""
                INSERT INTO MS_MARCO (query_id, query, query_type, answers, passages, well_formed_answers)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            """, (
                row['query_id'],
                row['query'],
                row['query_type'],
                answers_json,
                passages_json,
                well_formed_json
            ))
        
        conn.commit()
        print(f"Inserted {len(df)} row to database.")
        
    except Exception as e:
        print(f"Error: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    import_csv_to_database(csv_path)