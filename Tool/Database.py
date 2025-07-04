import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database configuration from environment variables with defaults
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
port = int(os.getenv("DB_PORT"))
name_dbname = os.getenv("DB_NAME")

def create_database_and_tables():
    db_params = {
        'initial_dbname': 'postgres',  
        'user': user,            
        'password': password,    
        'host': 'localhost',
        'port': port,            
        'new_dbname': name_dbname 
    }
    
    conn = psycopg2.connect(
        dbname=db_params['initial_dbname'],
        user=db_params['user'],
        password=db_params['password'],
        host=db_params['host'],
        port=db_params['port']
    )
    
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"CREATE DATABASE {db_params['new_dbname']}")
        print(f"Database '{db_params['new_dbname']}' created successfully.")
    except psycopg2.errors.DuplicateDatabase:
        print(f"Database '{db_params['new_dbname']}' already exists.")
    finally:
        cursor.close()
        conn.close()
    
    conn = psycopg2.connect(
        dbname=db_params['new_dbname'],
        user=db_params['user'],
        password=db_params['password'],
        host=db_params['host'],
        port=db_params['port']
    )
    
    cursor = conn.cursor()
    
    # Create Semantic_Grouping table with OIE_Triples
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Semantic_Grouping (
        id SERIAL PRIMARY KEY,
        query TEXT NOT NULL,  
        Original_Paragraph TEXT NOT NULL,
        Sentences JSONB NOT NULL,  
        Embedding_Vectors JSONB NOT NULL,  
        Semantic_Chunking JSONB NOT NULL,
        OIE_Triples JSONB,          -- Lưu trữ triples từ OpenIE theo nhóm câu
        OIE_Sentence_Groups JSONB   -- Lưu trữ cấu trúc nhóm câu và triples
    );
    """)
    
    # Create Semantic_Splitter table with OIE_Triples
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Semantic_Splitter (
        id SERIAL PRIMARY KEY,
        query TEXT NOT NULL,  
        Original_Paragraph TEXT NOT NULL,
        Sentences JSONB NOT NULL,  
        Embedding_Vectors JSONB NOT NULL,  
        Semantic_Chunking JSONB NOT NULL,
        OIE_Triples JSONB,          -- Lưu trữ triples từ OpenIE theo nhóm câu
        OIE_Sentence_Groups JSONB   -- Lưu trữ cấu trúc nhóm câu và triples
    );
    """)
    
    # Create Text_Splitter table with OIE_Triples
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Text_Splitter (
        id SERIAL PRIMARY KEY,
        query TEXT NOT NULL,  
        Original_Paragraph TEXT NOT NULL,
        Sentences JSONB NOT NULL,  
        Chunking JSONB NOT NULL,
        OIE_Triples JSONB,          -- Lưu trữ tất cả triples từ OpenIE
        OIE_Sentence_Groups JSONB   -- Lưu trữ cấu trúc nhóm câu và triples
    );
    """)
    
    # Create MS_MARCO table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS MS_MARCO (
        id SERIAL PRIMARY KEY,
        query_id TEXT NOT NULL,
        query TEXT NOT NULL,
        query_type TEXT,
        answers JSONB,  
        passages JSONB,  
        well_formed_answers JSONB  
    );
    """)
    
    conn.commit()
    print("Tables created successfully.")
    
    cursor.close()
    conn.close()

def connect_to_db():
    """Connect to the PostgreSQL database."""
    conn = psycopg2.connect(
        dbname="semantic_search_db",
        user=user,           
        password=password,   
        host="localhost",
        port=port            
    )
    return conn

if __name__ == "__main__":
    create_database_and_tables()