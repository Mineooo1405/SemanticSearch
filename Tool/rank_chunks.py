import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

try:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    sys.path.append(str(project_root))
except NameError:
    # Fallback for interactive environments
    project_root = Path(os.getcwd())
    sys.path.append(str(project_root))

from Sentence_Embedding import sentence_embedding
from rank_bm25 import BM25Okapi


def simple_tokenizer(text):

    if not isinstance(text, str):
        return []
    return text.lower().split()

def rank_by_bm25(query: str, chunks_df: pd.DataFrame, text_column: str = 'chunk_text') -> pd.DataFrame:

    print("\n--- Ranking with BM25 ---")
    if text_column not in chunks_df.columns:
        raise ValueError(f"Text column '{text_column}' not found in the DataFrame. Available columns: {chunks_df.columns.tolist()}")

    corpus = chunks_df[text_column].fillna("").tolist()
    tokenized_corpus = [simple_tokenizer(doc) for doc in corpus]
    
    print(f"Fitting BM25 on {len(tokenized_corpus)} documents...")
    # Use epsilon smoothing (default 0.25) so IDF never becomes negative, avoiding negative BM25 scores
    bm25 = BM25Okapi(tokenized_corpus, epsilon=0.25)
    
    print(f"Scoring query: '{query}'")
    tokenized_query = simple_tokenizer(query)
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Ensure scores are non-negative (BM25 may still yield small negatives even with epsilon smoothing)
    doc_scores = np.maximum(doc_scores, 0.0)
    
    ranked_df = chunks_df.copy()
    ranked_df['bm25_score'] = doc_scores
    
    # Sort by score in descending order
    ranked_df = ranked_df.sort_values(by='bm25_score', ascending=False).reset_index(drop=True)
    
    print(f"Ranking complete. Top 3 chunks:")
    print(ranked_df.head(3))
    
    return ranked_df

def rank_by_cosine_similarity(query: str, chunks_df: pd.DataFrame, text_column: str = 'chunk_text', model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32) -> pd.DataFrame:
    print("\n--- Ranking with Cosine Similarity ---")
    if text_column not in chunks_df.columns:
        raise ValueError(f"Text column '{text_column}' not found in the DataFrame. Available columns: {chunks_df.columns.tolist()}")
    
    corpus = chunks_df[text_column].fillna("").tolist()
    
    print(f"Embedding query: '{query}' using model '{model_name}'...")
    query_embedding = sentence_embedding(text_list=[query], model_name=model_name)
    
    if query_embedding is None or query_embedding.shape[0] == 0:
        raise RuntimeError("Failed to embed the query.")
        
    print(f"Embedding {len(corpus)} documents in batches of {batch_size}...")
    corpus_embeddings = sentence_embedding(text_list=corpus, model_name=model_name, batch_size=batch_size)
    
    if corpus_embeddings is None or corpus_embeddings.shape[0] == 0:
        raise RuntimeError("Failed to embed the corpus documents.")

    print("Calculating cosine similarity scores...")
    # Reshape query embedding for broadcasting
    query_embedding = query_embedding.reshape(1, -1)
    
    # Calculate scores
    cosine_scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
    
    ranked_df = chunks_df.copy()
    ranked_df['cosine_score'] = cosine_scores
    
    # Sort by score in descending order
    ranked_df = ranked_df.sort_values(by='cosine_score', ascending=False).reset_index(drop=True)
    
    print("Ranking complete. Top 3 chunks:")
    print(ranked_df.head(3))
    
    return ranked_df

# -------------------- NEW: Reciprocal Rank Fusion --------------------

def rank_by_rrf(cosine_df: pd.DataFrame, bm25_df: pd.DataFrame, k: int = 60, id_column: str = 'chunk_id') -> pd.DataFrame:
    """Combine Cosine and BM25 rankings using Reciprocal Rank Fusion (RRF).

    The RRF score for a document d is:
        score(d) = Σ_i 1 / (k + rank_i(d))
    where rank_i(d) is the rank position of document d in the i-th ranking list.

    Args:
        cosine_df: DataFrame already ranked by cosine similarity (higher score first).
        bm25_df:  DataFrame already ranked by BM25 (higher score first).
        k:        Constant that dampens the impact of lower-ranked documents (default 60).
        id_column:Column that uniquely identifies each chunk/document (default 'chunk_id').

    Returns:
        DataFrame with an added column `rrf_score`, sorted in descending order.
    """

    if id_column not in cosine_df.columns or id_column not in bm25_df.columns:
        raise ValueError(f"Column '{id_column}' must exist in both ranking DataFrames for RRF.")

    # Build rank lookup tables
    cosine_ranks = cosine_df[[id_column]].copy()
    cosine_ranks['rank_cosine'] = np.arange(1, len(cosine_ranks) + 1)

    bm25_ranks = bm25_df[[id_column]].copy()
    bm25_ranks['rank_bm25'] = np.arange(1, len(bm25_ranks) + 1)

    # Merge ranks (outer join to include docs that appear in only one list)
    merged_ranks = pd.merge(cosine_ranks, bm25_ranks, on=id_column, how='outer')

    # Fill missing ranks with a large value to give them minimal influence
    max_rank = max(len(cosine_ranks), len(bm25_ranks)) + k
    merged_ranks['rank_cosine'] = merged_ranks['rank_cosine'].fillna(max_rank)
    merged_ranks['rank_bm25'] = merged_ranks['rank_bm25'].fillna(max_rank)

    # Calculate RRF score
    merged_ranks['rrf_score'] = 1.0 / (k + merged_ranks['rank_cosine']) + 1.0 / (k + merged_ranks['rank_bm25'])

    # Attach RRF score và giữ cả cosine_score, bm25_score
    # Bắt đầu với outer-merge để bảo toàn mọi cột
    rrf_df = pd.merge(bm25_df, cosine_df[[id_column, 'cosine_score']], on=id_column, how='outer')
    rrf_df = pd.merge(rrf_df, merged_ranks[[id_column, 'rrf_score']], on=id_column, how='left')

    # Sort by RRF score (higher is better)
    rrf_df = rrf_df.sort_values(by='rrf_score', ascending=False).reset_index(drop=True)

    print("\n--- Ranking with Reciprocal Rank Fusion (RRF) ---")
    print("Ranking complete. Top 3 chunks:")
    print(rrf_df.head(3))

    return rrf_df

def interactive_mode():

    print("\n--- Interactive Chunk Ranker ---")
    
    # 1. Get input file path
    input_file = ""
    while True:
        try:
            path_str = input("Enter the path to your input TSV file: ")
            input_path = Path(path_str.strip())
            if input_path.is_file():
                input_file = str(input_path)
                break
            else:
                print(f"Error: File not found at '{path_str}'. Please try again.")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # 2. Load a sample of the DF to inspect columns
    try:
        df = pd.read_csv(input_file, sep='\t', on_bad_lines='warn', nrows=5)
        df.columns = df.columns.str.strip()
        print("\nFile loaded. Found columns:")
        print(" | ".join(df.columns.tolist()))
    except Exception as e:
        print(f"Error reading file columns: {e}")
        sys.exit(1)
        
    # 3. Get column names from user
    print("\nPlease confirm the column names to use (press Enter to accept default).")
    query_id_column = input("Query ID column (default: query_id): ") or "query_id"
    query_text_column = input("Query text column (default: query_text): ") or "query_text"
    text_column = input("Chunk text column to rank (default: chunk_text): ") or "chunk_text"
    
    # 4. Get other parameters
    output_dir = input("Output directory (default: ./ranking_results): ") or "./ranking_results"
    embedding_model = input("Embedding model (default: all-MiniLM-L6-v2): ") or "all-MiniLM-L6-v2"
    
    # 4b. Hỏi ngưỡng (tùy chọn)
    def _ask_float(prompt: str, default: float):
        try:
            val = input(prompt) or str(default)
            return float(val)
        except ValueError:
            return default

    cosine_pos_thr = _ask_float("Cosine POS threshold (default 0.75): ", 0.75)
    cosine_neg_thr = _ask_float("Cosine NEG threshold (default 0.25): ", 0.25)
    bm25_pos_thr   = _ask_float("BM25 POS threshold   (default 3.0): ", 3.0)
    bm25_neg_thr   = _ask_float("BM25 NEG threshold   (default 1.0): ", 1.0)

    # 5. Return a namespace object similar to argparse
    class Args:
        def __init__(self):
            self.input_file = ""
            self.query_id_column = ""
            self.query_text_column = ""
            self.text_column = ""
            self.output_dir = ""
            self.embedding_model = ""
            self.cosine_pos_threshold = 0.75
            self.cosine_neg_threshold = 0.25
            self.bm25_pos_threshold = 3.0
            self.bm25_neg_threshold = 1.0

    args = Args()
    args.input_file = input_file
    args.query_id_column = query_id_column
    args.query_text_column = query_text_column
    args.text_column = text_column
    args.output_dir = output_dir
    args.embedding_model = embedding_model
    args.cosine_pos_threshold = cosine_pos_thr
    args.cosine_neg_threshold = cosine_neg_thr
    args.bm25_pos_threshold = bm25_pos_thr
    args.bm25_neg_threshold = bm25_neg_thr
    
    print("\nConfiguration complete. Starting ranking process...")
    return args

def main():
    """
    Main function to parse arguments and run the ranking processes.
    """
    parser = argparse.ArgumentParser(
        description="Rank chunks within each query group. Run without arguments for interactive mode."
    )
    # Make input_file optional (nargs='?') to allow for interactive mode
    parser.add_argument("input_file", type=str, nargs='?', default=None, 
                        help="Path to the input TSV file. If not provided, the script will enter interactive mode.")
    parser.add_argument("--query_id_column", type=str, default="query_id", help="Name of the column with the query identifier.")
    parser.add_argument("--query_text_column", type=str, default="query_text", help="Name of the column with the query text.")
    parser.add_argument("--text_column", type=str, default="chunk_text", help="Name of the column with the chunk text to rank.")
    parser.add_argument("--output_dir", type=str, default="./ranking_results", help="Directory to save the ranked output files.")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model for cosine similarity.")
    parser.add_argument("--cosine_pos_threshold", type=float, default=0.75, help="Upper threshold: cosine_score >= value → positive")
    parser.add_argument("--cosine_neg_threshold", type=float, default=0.25, help="Lower threshold: cosine_score <= value → negative")
    parser.add_argument("--bm25_pos_threshold", type=float, default=3.0, help="Upper threshold: bm25_score >= value → positive")
    parser.add_argument("--bm25_neg_threshold", type=float, default=1.0, help="Lower threshold: bm25_score <= value → negative")

    args = parser.parse_args()

    # If input_file is not provided via command line, launch interactive mode
    if args.input_file is None:
        args = interactive_mode()

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: Input file not found at {args.input_file}")
        sys.exit(1)
        
    print(f"Loading data from {args.input_file}...")
    try:
        chunks_df = pd.read_csv(input_path, sep='\t', on_bad_lines='warn')
        chunks_df.columns = chunks_df.columns.str.strip()
        print(f"Successfully loaded {len(chunks_df)} rows.")
        print(f"Columns found: {chunks_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading TSV file: {e}")
        sys.exit(1)

    # Validate required columns
    required_cols = [args.query_id_column, args.query_text_column, args.text_column]
    for col in required_cols:
        if col not in chunks_df.columns:
            print(f"Error: Required column '{col}' not found in the input file.")
            sys.exit(1)

    # --- Group by Query ID and Rank ---
    all_cosine_ranked = []
    all_bm25_ranked = []
    all_rrf_ranked = []  # NEW: store RRF results

    # Lists to collect labeled data
    cosine_labeled_rows = []
    bm25_labeled_rows = []

    query_groups = chunks_df.groupby(args.query_id_column)
    print(f"\nFound {len(query_groups)} unique queries. Starting ranking for each group...")

    for query_id, group_df in query_groups:
        print(f"\n--- Processing Query ID: {query_id} ({len(group_df)} chunks) ---")
        
        # Get the query text (should be the same for the whole group)
        query_text = group_df[args.query_text_column].iloc[0]
        
        # Method 1: Cosine Similarity
        try:
            print(f"Ranking with Cosine Similarity for query: '{query_text[:100]}...'")
            cosine_ranked_group = rank_by_cosine_similarity(query_text, group_df, text_column=args.text_column, model_name=args.embedding_model)
            all_cosine_ranked.append(cosine_ranked_group)

            # Filter pos / neg theo upper/lower threshold
            pos_rows = cosine_ranked_group[cosine_ranked_group['cosine_score'] >= args.cosine_pos_threshold].copy()
            neg_rows = cosine_ranked_group[cosine_ranked_group['cosine_score'] <= args.cosine_neg_threshold].copy()

            pos_rows['label'] = 1
            neg_rows['label'] = 0

            if not pos_rows.empty:
                cosine_labeled_rows.append(pos_rows)
            if not neg_rows.empty:
                cosine_labeled_rows.append(neg_rows)
        except Exception as e:
            print(f"  > Error during Cosine Similarity ranking for query '{query_id}': {e}")

        # Method 2: BM25
        try:
            print(f"Ranking with BM25 for query: '{query_text[:100]}...'")
            bm25_ranked_group = rank_by_bm25(query_text, group_df, text_column=args.text_column)
            all_bm25_ranked.append(bm25_ranked_group)

            # Filter pos / neg theo upper/lower threshold
            bm25_pos_rows = bm25_ranked_group[bm25_ranked_group['bm25_score'] >= args.bm25_pos_threshold].copy()
            bm25_neg_rows = bm25_ranked_group[bm25_ranked_group['bm25_score'] <= args.bm25_neg_threshold].copy()

            bm25_pos_rows['label'] = 1
            bm25_neg_rows['label'] = 0

            if not bm25_pos_rows.empty:
                bm25_labeled_rows.append(bm25_pos_rows)
            if not bm25_neg_rows.empty:
                bm25_labeled_rows.append(bm25_neg_rows)
        except Exception as e:
            print(f"  > Error during BM25 ranking for query '{query_id}': {e}")

        # Method 3: Reciprocal Rank Fusion (requires both lists)
        try:
            print("Combining rankings with Reciprocal Rank Fusion (RRF)...")
            rrf_ranked_group = rank_by_rrf(cosine_ranked_group, bm25_ranked_group, k=60)
            all_rrf_ranked.append(rrf_ranked_group)
        except Exception as e:
            print(f"  > Error during RRF ranking for query '{query_id}': {e}")
            
    # --- Consolidate and Save Results ---
    if all_cosine_ranked:
        final_cosine_df = pd.concat(all_cosine_ranked).reset_index(drop=True)
        cosine_output_path = output_dir / f"{input_path.stem}_ranked_cosine.tsv"
        final_cosine_df.to_csv(cosine_output_path, sep='\t', index=False)
        print(f"\nAll Cosine Similarity results saved to: {cosine_output_path}")

        # Lưu dataset có nhãn
        if cosine_labeled_rows:
            cosine_labeled_df = pd.concat(cosine_labeled_rows).reset_index(drop=True)
            pos_df = cosine_labeled_df[cosine_labeled_df['label'] == 1].sort_values(by='cosine_score', ascending=False)  # type: ignore[arg-type]
            neg_df = cosine_labeled_df[cosine_labeled_df['label'] == 0].sort_values(by='cosine_score', ascending=True)   # type: ignore[arg-type]

            pos_path = output_dir / f"{input_path.stem}_cosine_pos.tsv"
            neg_path = output_dir / f"{input_path.stem}_cosine_neg.tsv"
            pos_df.to_csv(pos_path, sep='\t', index=False)
            neg_df.to_csv(neg_path, sep='\t', index=False)
            print(f"Cosine datasets saved to: {pos_path} (pos) & {neg_path} (neg)")

    if all_bm25_ranked:
        final_bm25_df = pd.concat(all_bm25_ranked).reset_index(drop=True)
        bm25_output_path = output_dir / f"{input_path.stem}_ranked_bm25.tsv"
        final_bm25_df.to_csv(bm25_output_path, sep='\t', index=False)
        print(f"All BM25 results saved to: {bm25_output_path}")

        # Lưu dataset có nhãn
        if bm25_labeled_rows:
            bm25_labeled_df = pd.concat(bm25_labeled_rows).reset_index(drop=True)
            pos_df = bm25_labeled_df[bm25_labeled_df['label'] == 1].sort_values(by='bm25_score', ascending=False)  # type: ignore[arg-type]
            neg_df = bm25_labeled_df[bm25_labeled_df['label'] == 0].sort_values(by='bm25_score', ascending=True)   # type: ignore[arg-type]

            pos_path = output_dir / f"{input_path.stem}_bm25_pos.tsv"
            neg_path = output_dir / f"{input_path.stem}_bm25_neg.tsv"
            pos_df.to_csv(pos_path, sep='\t', index=False)
            neg_df.to_csv(neg_path, sep='\t', index=False)
            print(f"BM25 datasets saved to: {pos_path} (pos) & {neg_path} (neg)")

    # --- Save RRF Results ---
    if all_rrf_ranked:
        final_rrf_df = pd.concat(all_rrf_ranked).reset_index(drop=True)
        rrf_output_path = output_dir / f"{input_path.stem}_ranked_rrf.tsv"
        final_rrf_df.to_csv(rrf_output_path, sep='\t', index=False)
        print(f"All RRF results saved to: {rrf_output_path}")


if __name__ == "__main__":
    main() 