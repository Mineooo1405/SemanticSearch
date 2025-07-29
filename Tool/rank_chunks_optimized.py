"""
Optimized ranking module with memory efficiency and caching
"""
import argparse
import pandas as pd
import numpy as np
import gc
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Quick helper: estimate file size & memory usage (was removed earlier)
# ---------------------------------------------------------------------------

def estimate_memory_usage(file_path: str) -> Dict[str, float]:
    """Return rough memory usage estimates for a TSV file."""
    if not Path(file_path).exists():
        return {"error": "File not found"}

    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Assume pandas expands 4× in RAM, embedding adds ~70 %
    pandas_overhead_factor = 4.0
    embedding_overhead_factor = 1.7

    estimated_memory_gb = (file_size_bytes * pandas_overhead_factor) / (1024 ** 3)
    peak_memory_gb = estimated_memory_gb * embedding_overhead_factor

    recommended_chunk_size = max(10000, min(100000, int(50000 / max(1, estimated_memory_gb / 2))))

    return {
        "file_size_mb": file_size_mb,
        "estimated_memory_gb": estimated_memory_gb,
        "peak_memory_gb": peak_memory_gb,
        "recommended_chunk_size": recommended_chunk_size,
    }

# ---------------------------------------------------------------------------
# Utility: detect and normalize column names
# ---------------------------------------------------------------------------

_QUERY_TEXT_KEYS = {"query_text", "query", "question"}
_CHUNK_TEXT_KEYS = {"chunk_text", "passage", "text"}
_QUERY_ID_KEYS   = {"query_id", "qid"}
_CHUNK_ID_KEYS   = {"chunk_id", "cid", "pid"}
_LABEL_KEYS      = {"label", "score", "target"}


def _find_col(existing: list[str], candidates: set[str]) -> Optional[str]:
    for col in existing:
        if col.lower() in candidates:
            return col
    return None


def _standardize_columns(df: pd.DataFrame, *, require_query_text: bool = True) -> pd.DataFrame:
    """Chuẩn hoá tên cột về dạng chuẩn.

    Args:
        df: DataFrame input.
        require_query_text: Nếu True, thiếu query_text sẽ raise; nếu False cho phép thiếu
            (sẽ được bổ sung sau).
    """
    existing = [c.strip() for c in df.columns]

    mapping: dict[str, str] = {}

    qtext = _find_col(existing, _QUERY_TEXT_KEYS)
    ctext = _find_col(existing, _CHUNK_TEXT_KEYS)

    # ensure chunk_text mapping exists
    mapping[ctext] = "chunk_text"

    if qtext:
        mapping[qtext] = "query_text"

    qid = _find_col(existing, _QUERY_ID_KEYS)
    if qid:
        mapping[qid] = "query_id"

    cid = _find_col(existing, _CHUNK_ID_KEYS)
    if cid:
        mapping[cid] = "chunk_id"

    lab = _find_col(existing, _LABEL_KEYS)
    if lab:
        mapping[lab] = "label"

    df = df.rename(columns=mapping)
    return df


class OptimizedRanker:
    """Memory-efficient ranking with caching and batch processing"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device_preference: str = "dml", cache_size: int = 1000):
        self.model_name = model_name
        self.device_preference = device_preference if device_preference else "dml"  # Default to DirectML
        self.cache_size = cache_size
        
        # Caches for embeddings to avoid recomputation
        self.query_embedding_cache: Dict[str, np.ndarray] = {}
        self.chunk_embedding_cache: Dict[str, np.ndarray] = {}
        
        # Import here to avoid circular imports
        try:
            from Tool.Sentence_Embedding import sentence_embedding
            self.sentence_embedding = sentence_embedding
        except ImportError:
            from Sentence_Embedding import sentence_embedding
            self.sentence_embedding = sentence_embedding
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _manage_cache_size(self):
        """Keep cache size under limit"""
        if len(self.chunk_embedding_cache) > self.cache_size:
            # Remove oldest 25% of entries
            to_remove = len(self.chunk_embedding_cache) // 4
            keys_to_remove = list(self.chunk_embedding_cache.keys())[:to_remove]
            for key in keys_to_remove:
                del self.chunk_embedding_cache[key]
            gc.collect()
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding with caching"""
        query_hash = self._get_text_hash(query)
        
        if query_hash in self.query_embedding_cache:
            return self.query_embedding_cache[query_hash]
        
        # Compute embedding
        embedding = self.sentence_embedding(
            text_list=[query], 
            model_name=self.model_name, 
            device_preference=self.device_preference
        )
        
        if embedding is not None and embedding.shape[0] > 0:
            self.query_embedding_cache[query_hash] = embedding[0]
            return embedding[0]
        else:
            raise RuntimeError(f"Failed to embed query: {query}")
    
    def get_chunk_embeddings_batch(self, chunks: List[str], batch_size: int = 32) -> np.ndarray:
        """Get chunk embeddings with caching and batch processing"""
        embeddings = []
        uncached_indices = []
        uncached_chunks = []
        
        # Check cache first
        for i, chunk in enumerate(chunks):
            chunk_hash = self._get_text_hash(chunk)
            if chunk_hash in self.chunk_embedding_cache:
                embeddings.append(self.chunk_embedding_cache[chunk_hash])
            else:
                embeddings.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_chunks.append(chunk)
        
        # Batch process uncached chunks
        if uncached_chunks:
            print(f"Computing embeddings for {len(uncached_chunks)} uncached chunks...")
            new_embeddings = self.sentence_embedding(
                text_list=uncached_chunks,
                model_name=self.model_name,
                batch_size=batch_size,
                device_preference=self.device_preference
            )
            
            if new_embeddings is not None and new_embeddings.shape[0] == len(uncached_chunks):
                # Cache new embeddings and fill placeholders
                for i, (orig_idx, chunk) in enumerate(zip(uncached_indices, uncached_chunks)):
                    chunk_hash = self._get_text_hash(chunk)
                    self.chunk_embedding_cache[chunk_hash] = new_embeddings[i]
                    embeddings[orig_idx] = new_embeddings[i]
            else:
                raise RuntimeError("Failed to embed some chunks")
        
        # Manage cache size
        self._manage_cache_size()
        
        return np.array(embeddings)
    
    def rank_single_query_optimized(self, query: str, chunks_df: pd.DataFrame, 
                                  text_column: str = 'chunk_text', id_column: str = 'chunk_id') -> pd.DataFrame:
        """Optimized ranking for a single query with caching"""
        
        if text_column not in chunks_df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        chunks = chunks_df[text_column].fillna("").tolist()
        
        # Get embeddings (with caching)
        query_embedding = self.get_query_embedding(query)
        chunk_embeddings = self.get_chunk_embeddings_batch(chunks)
        
        # Calculate cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        cosine_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # BM25 scoring (fast, no caching needed)
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks, epsilon=0.25)
        tokenized_query = query.lower().split()
        bm25_scores = np.maximum(bm25.get_scores(tokenized_query), 0.0)
        
        # RRF fusion
        cosine_ranks = np.argsort(-cosine_scores) + 1  # Higher score = lower rank number
        bm25_ranks = np.argsort(-bm25_scores) + 1
        
        # Create rank lookup
        cosine_rank_lookup = np.zeros(len(chunks))
        bm25_rank_lookup = np.zeros(len(chunks))
        
        for rank, idx in enumerate(np.argsort(-cosine_scores)):
            cosine_rank_lookup[idx] = rank + 1
        for rank, idx in enumerate(np.argsort(-bm25_scores)):
            bm25_rank_lookup[idx] = rank + 1
        
        # Calculate RRF scores
        k = 60
        rrf_scores = 1.0 / (k + cosine_rank_lookup) + 1.0 / (k + bm25_rank_lookup)
        
        # Create result dataframe
        result_df = chunks_df.copy()
        result_df['cosine_score'] = cosine_scores
        result_df['bm25_score'] = bm25_scores
        result_df['rrf_score'] = rrf_scores
        
        # Sort by RRF score
        result_df = result_df.sort_values(by='rrf_score', ascending=False).reset_index(drop=True)
        
        return result_df


def rank_and_filter_chunks_optimized(chunks_tsv: str, output_dir: Path,
                                    original_tsv: str,
                                    upper_percentile: int = 80, lower_percentile: int = 20,
                                    model_name: str = "thenlper/gte-base", max_workers: int = None,
                                    chunk_size: int = 50000) -> str:
    """Memory-optimized version with chunked data loading to prevent RAM overflow
    
    Args:
        chunks_tsv: Path to input TSV file
        output_dir: Output directory
        upper_percentile: Upper percentile for positive samples
        lower_percentile: Lower percentile for negative samples
        model_name: Embedding model name
        max_workers: Number of parallel workers (None = sequential processing)
        chunk_size: Number of rows to process at once (default: 50,000)
    """
    
    if not Path(chunks_tsv).exists():
        return ""
    
    print(f"Starting memory-optimized ranking from {chunks_tsv}...")
    
    # ------------------------------------------------------------
    # Load mapping query_id → query_text từ file original
    # ------------------------------------------------------------
    try:
        mapping_df = pd.read_csv(original_tsv, sep='\t', usecols=['query_id', 'query_text'], engine='python')
        query_map = dict(mapping_df.groupby('query_id')['query_text'].first())
        del mapping_df
    except Exception as e:
        print(f"Error loading original TSV for query_text mapping: {e}")
        return ""

    # Estimate memory requirements and adjust chunk size accordingly
    memory_estimate = estimate_memory_usage(chunks_tsv)
    if "error" in memory_estimate:
        print(f"Warning: Could not estimate memory usage: {memory_estimate['error']}")
        file_size_mb = Path(chunks_tsv).stat().st_size / (1024 * 1024)
    else:
        file_size_mb = memory_estimate["file_size_mb"]
        estimated_memory = memory_estimate["estimated_memory_gb"]
        peak_memory = memory_estimate["peak_memory_gb"]
        recommended_chunk = memory_estimate["recommended_chunk_size"]
        
        print(f"Memory analysis:")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Estimated memory usage: {estimated_memory:.1f} GB")
        print(f"  Peak memory usage: {peak_memory:.1f} GB")
        print(f"  Recommended chunk size: {recommended_chunk:,} rows")
        
        # Use smaller chunk size if file is very large
        if chunk_size > recommended_chunk:
            chunk_size = recommended_chunk
            print(f"  Adjusting chunk size to {chunk_size:,} rows for memory efficiency")
    
    print(f"Input file size: {file_size_mb:.1f} MB")
    
    # First pass: Count total rows and get column info without loading all data
    print("Scanning file structure...")
    try:
        header_sample = pd.read_csv(chunks_tsv, sep='\t', nrows=5, on_bad_lines='warn', engine='python')
        header_sample = _standardize_columns(header_sample, require_query_text=False)
        required_cols = {'chunk_text'}
        if not required_cols.issubset(set(header_sample.columns)):
            print("File thiếu cột bắt buộc (chunk_text)")
            return ""
        
        # Get total row count efficiently
        total_rows = sum(1 for _ in open(chunks_tsv, 'r', encoding='utf-8')) - 1  # Subtract header
        print(f"Total rows: {total_rows:,}")
        
    except Exception as e:
        print(f"Error scanning file: {e}")
        return ""
    
    # Process data in memory-friendly chunks
    kept_rows = []
    processed_rows = 0
    
    print(f"Processing in chunks of {chunk_size:,} rows to optimize memory usage...")
    
    # Monitor memory usage during processing
    try:
        import psutil
        initial_memory = psutil.virtual_memory()
        print(f"Initial RAM usage: {(initial_memory.total - initial_memory.available) / (1024**3):.1f}GB / {initial_memory.total / (1024**3):.1f}GB")
    except ImportError:
        print("Memory monitoring not available (psutil not installed)")
    
    # Read and process file in chunks
    chunk_iterator = pd.read_csv(chunks_tsv, sep='\t', chunksize=chunk_size, 
                                on_bad_lines='warn', engine='python')
    
    for chunk_num, df_chunk in enumerate(chunk_iterator, 1):
        try:
            df_chunk.columns = df_chunk.columns.str.strip()
            df_chunk = _standardize_columns(df_chunk, require_query_text=False)

            # Thêm query_text nếu chưa có
            if 'query_text' not in df_chunk.columns and 'query_id' in df_chunk.columns:
                df_chunk['query_text'] = df_chunk['query_id'].map(query_map).fillna('')
            
            processed_rows += len(df_chunk)
            
            print(f"Processing chunk {chunk_num}: rows {processed_rows - len(df_chunk) + 1:,} to {processed_rows:,}")
            
            # Get unique queries in this chunk
            chunk_queries = df_chunk['query_id'].nunique()
            print(f"  Found {chunk_queries} unique queries in chunk {chunk_num}")
            
            # Log memory usage every few chunks
            if chunk_num % 3 == 0:
                try:
                    current_memory = psutil.virtual_memory()
                    memory_used = (current_memory.total - current_memory.available) / (1024**3)
                    print(f"  Current RAM usage: {memory_used:.1f}GB / {current_memory.total / (1024**3):.1f}GB ({current_memory.percent:.1f}%)")
                except:
                    pass
            
            # Process this chunk based on max_workers
            if max_workers and max_workers > 1:
                chunk_results = _process_queries_parallel(df_chunk, model_name, upper_percentile, 
                                                        lower_percentile, max_workers)
            else:
                chunk_results = _process_queries_sequential(df_chunk, model_name, upper_percentile, 
                                                          lower_percentile)
            
            if chunk_results:
                kept_rows.extend(chunk_results)
                print(f"  Chunk {chunk_num} yielded {len(chunk_results)} ranked query groups")
            
            # Force garbage collection after each chunk
            del df_chunk
            gc.collect()
            
        except Exception as e:
            print(f"Error processing chunk {chunk_num}: {e}")
            continue
    
    print(f"Completed processing {processed_rows:,} total rows in {chunk_num} chunks")
    
    if not kept_rows:
        print("No data to save after filtering")
        return ""
    
    # Combine results with memory optimization
    print(f"Combining {len(kept_rows)} result groups...")
    
    # Process results in smaller batches to avoid memory spike
    batch_size = 10  # Process 10 result DataFrames at a time
    final_dfs = []
    
    for i in range(0, len(kept_rows), batch_size):
        batch = kept_rows[i:i+batch_size]
        if batch:
            batch_df = pd.concat(batch, ignore_index=True)
            final_dfs.append(batch_df)
            
            # Clear the batch from memory
            for df in batch:
                del df
            del batch
            gc.collect()
            
            print(f"  Processed batch {(i//batch_size) + 1}/{(len(kept_rows) + batch_size - 1) // batch_size}")
    
    # Final combine
    print("Final combination...")
    final_df = pd.concat(final_dfs, ignore_index=True)
    
    # Clear intermediate results
    del final_dfs, kept_rows
    gc.collect()
    
    print(f"Combined dataset: {len(final_df):,} rows")
    
    # Save main file
    save_path = output_dir / f"{Path(chunks_tsv).stem}_rrf_filtered.tsv"
    final_df.to_csv(save_path, sep='\t', index=False)
    
    # Save 3-column version
    try:
        simple_df = final_df[['query_text', 'chunk_text', 'label']].copy()
        simple_df.columns = ['query', 'passage', 'label']
        simple_path = output_dir / f"{Path(chunks_tsv).stem}_rrf_filtered_3col.tsv"
        simple_df.to_csv(simple_path, sep='\t', index=False)
        del simple_df  # Free memory immediately
    except Exception as e:
        print(f"Warning: Could not create 3-column file: {e}")
    
    print(f"Memory-optimized ranking complete. Saved {len(final_df):,} filtered chunks to {save_path}")
    
    # Final cleanup
    del final_df
    gc.collect()
    
    return str(save_path)


def _process_queries_sequential(df: pd.DataFrame, model_name: str, 
                              upper_percentile: int, lower_percentile: int) -> List[pd.DataFrame]:
    """Sequential processing with single model instance"""
    ranker = OptimizedRanker(model_name=model_name)
    kept_rows = []
    processed_queries = 0
    total_queries = df['query_id'].nunique()
    
    # Process each query group
    for query_id, group_df in df.groupby('query_id'):
        try:
            processed_queries += 1
            if processed_queries % 10 == 0:
                print(f"Sequential: Processed {processed_queries}/{total_queries} queries...")
                gc.collect()  # Periodic garbage collection
            
            query_text = str(group_df['query_text'].iloc[0])
            
            # Skip very small groups
            if len(group_df) < 2:
                continue
            
            # Use optimized ranking
            ranked_df = ranker.rank_single_query_optimized(query_text, group_df)
            
            if ranked_df.empty:
                continue
            
            # Apply percentile filtering
            scores = ranked_df['rrf_score'].to_numpy(dtype=float)
            pos_thr = np.percentile(scores, upper_percentile)
            neg_thr = np.percentile(scores, lower_percentile)
            
            selected = ranked_df[(ranked_df['rrf_score'] >= pos_thr) | 
                               (ranked_df['rrf_score'] <= neg_thr)]
            
            if not selected.empty:
                selected['label'] = (selected['rrf_score'] >= pos_thr).astype(int)
                kept_rows.append(selected)
            
        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            continue
    
    # Clean up
    del ranker
    gc.collect()
    
    return kept_rows


def _process_queries_parallel(df: pd.DataFrame, model_name: str, 
                            upper_percentile: int, lower_percentile: int, max_workers: int) -> List[pd.DataFrame]:
    """Parallel processing with dedicated model instances per worker"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    # Limit workers to available resources
    effective_workers = min(max_workers, multiprocessing.cpu_count(), df['query_id'].nunique())
    print(f"Using {effective_workers} effective workers for ranking")
    
    # Group queries for batch processing
    query_groups = []
    for query_id, group_df in df.groupby('query_id'):
        if len(group_df) >= 2:  # Skip very small groups
            query_groups.append((query_id, group_df))
    
    if not query_groups:
        return []
    
    print(f"Processing {len(query_groups)} query groups with {effective_workers} workers...")
    
    # Process in parallel with dedicated models per worker
    kept_rows = []
    
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        # Submit all tasks
        future_to_query = {}
        worker_counter = 0
        
        for query_id, group_df in query_groups:
            worker_id = worker_counter % effective_workers + 1
            future = executor.submit(
                _rank_single_query_worker,
                query_id,
                group_df,
                model_name,
                upper_percentile,
                lower_percentile,
                worker_id
            )
            future_to_query[future] = query_id
            worker_counter += 1
        
        # Collect results
        completed_count = 0
        for future in as_completed(future_to_query):
            query_id = future_to_query[future]
            completed_count += 1
            
            try:
                result = future.result(timeout=300)  # 5 minute timeout per query
                if result is not None and not result.empty:
                    kept_rows.append(result)
                
                if completed_count % 10 == 0:
                    print(f"Parallel: Completed {completed_count}/{len(query_groups)} queries...")
                    
            except Exception as e:
                print(f"Error processing query {query_id} in parallel: {e}")
                continue
    
    print(f"Parallel processing completed. {len(kept_rows)} successful query rankings.")
    return kept_rows


def _rank_single_query_worker(query_id: str, group_df: pd.DataFrame, model_name: str,
                            upper_percentile: int, lower_percentile: int, worker_id: int) -> Optional[pd.DataFrame]:
    """Worker function for parallel query ranking with dedicated model
    
    Args:
        query_id: Query ID for logging
        group_df: DataFrame with chunks for this query
        model_name: Embedding model name
        upper_percentile: Upper percentile threshold
        lower_percentile: Lower percentile threshold  
        worker_id: Worker ID for logging
        
    Returns:
        Filtered DataFrame or None if error
    """
    try:
        # Each worker gets its own model instance
        ranker = OptimizedRanker(model_name=model_name)
        
        query_text = str(group_df['query_text'].iloc[0])
        
        # Use optimized ranking
        ranked_df = ranker.rank_single_query_optimized(query_text, group_df)
        
        if ranked_df.empty:
            return None
        
        # Apply percentile filtering
        scores = ranked_df['rrf_score'].to_numpy(dtype=float)
        pos_thr = np.percentile(scores, upper_percentile)
        neg_thr = np.percentile(scores, lower_percentile)
        
        selected = ranked_df[(ranked_df['rrf_score'] >= pos_thr) | 
                           (ranked_df['rrf_score'] <= neg_thr)].copy()  # Use copy() to avoid SettingWithCopyWarning
        
        if not selected.empty:
            selected['label'] = (selected['rrf_score'] >= pos_thr).astype(int)
            return selected
        else:
            return None
            
    except Exception as e:
        print(f"Worker W{worker_id} error processing query {query_id}: {e}")
        return None
    finally:
        # Cleanup worker resources
        try:
            del ranker
        except:
            pass
        gc.collect()


# Legacy compatibility functions
def rank_by_cosine_similarity(query: str, chunks_df: pd.DataFrame, text_column: str = 'chunk_text', 
                            model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32, 
                            *, device_preference: str = None) -> pd.DataFrame:
    """Legacy function maintained for compatibility"""
    ranker = OptimizedRanker(model_name=model_name, device_preference=device_preference)
    result = ranker.rank_single_query_optimized(query, chunks_df, text_column)
    return result[['chunk_id', 'cosine_score'] + [col for col in chunks_df.columns if col != 'chunk_id']].sort_values('cosine_score', ascending=False)


def rank_by_bm25(query: str, chunks_df: pd.DataFrame, text_column: str = 'chunk_text') -> pd.DataFrame:
    """Legacy BM25 ranking function"""
    corpus = chunks_df[text_column].fillna("").tolist()
    tokenized_corpus = [text.lower().split() for text in corpus]
    
    bm25 = BM25Okapi(tokenized_corpus, epsilon=0.25)
    tokenized_query = query.lower().split()
    scores = np.maximum(bm25.get_scores(tokenized_query), 0.0)
    
    result_df = chunks_df.copy()
    result_df['bm25_score'] = scores
    return result_df.sort_values('bm25_score', ascending=False).reset_index(drop=True)


def rank_by_rrf(cosine_df: pd.DataFrame, bm25_df: pd.DataFrame, k: int = 60, 
               id_column: str = 'chunk_id') -> pd.DataFrame:
    """Legacy RRF function maintained for compatibility"""
    # Implementation same as before but cleaned up
    if id_column not in cosine_df.columns or id_column not in bm25_df.columns:
        raise ValueError(f"Column '{id_column}' must exist in both DataFrames")
    
    cosine_ranks = cosine_df[[id_column]].copy()
    cosine_ranks['rank_cosine'] = np.arange(1, len(cosine_ranks) + 1)
    
    bm25_ranks = bm25_df[[id_column]].copy()
    bm25_ranks['rank_bm25'] = np.arange(1, len(bm25_ranks) + 1)
    
    merged_ranks = pd.merge(cosine_ranks, bm25_ranks, on=id_column, how='outer')
    max_rank = max(len(cosine_ranks), len(bm25_ranks)) + k
    merged_ranks['rank_cosine'] = merged_ranks['rank_cosine'].fillna(max_rank)
    merged_ranks['rank_bm25'] = merged_ranks['rank_bm25'].fillna(max_rank)
    
    merged_ranks['rrf_score'] = (1.0 / (k + merged_ranks['rank_cosine']) + 
                                1.0 / (k + merged_ranks['rank_bm25']))
    
    rrf_df = pd.merge(bm25_df, cosine_df[[id_column, 'cosine_score']], on=id_column, how='outer')
    rrf_df = pd.merge(rrf_df, merged_ranks[[id_column, 'rrf_score']], on=id_column, how='left')
    
    return rrf_df.sort_values('rrf_score', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli():
    parser = argparse.ArgumentParser(description="Optimized RRF ranking for chunk TSV")
    parser.add_argument("input_tsv", type=Path, help="Đường dẫn file TSV chứa chunk")
    parser.add_argument("output_dir", type=Path, help="Thư mục lưu kết quả")
    parser.add_argument("original", type=Path, help="Đường dẫn file TSV gốc chứa query_text")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Tên mô hình embedding (mặc định MiniLM-L6)")
    parser.add_argument("--workers", type=int, default=4, help="Số worker song song cho ranking")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Số dòng đọc mỗi lượt")
    parser.add_argument("--up", type=int, default=80, help="Upper percentile POS")
    parser.add_argument("--low", type=int, default=20, help="Lower percentile NEG")
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    rank_and_filter_chunks_optimized(
        chunks_tsv=str(args.input_tsv),
        output_dir=args.output_dir,
        original_tsv=str(args.original),
        upper_percentile=args.up,
        lower_percentile=args.low,
        model_name=args.model,
        max_workers=args.workers,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _cli()
    else:
        # ----- Interactive prompt -----
        print("Interactive RRF Ranking UI")
        chunks_tsv = input("Enter path to CHUNKS TSV (from simple_chunk_controller): ").strip()
        original_tsv = input("Enter path to ORIGINAL TSV (query_id\tquery_text…): ").strip()
        output_dir = input("Enter output directory [training_datasets]: ").strip() or "training_datasets"

        model = input("Embedding model [sentence-transformers/all-MiniLM-L6-v2]: ").strip() or "sentence-transformers/all-MiniLM-L6-v2"
        workers = int(input("Parallel workers for ranking [4]: ") or "4")
        up = int(input("Upper percentile POS [80]: ") or "80")
        low = int(input("Lower percentile NEG [20]: ") or "20")
        chunk_sz = int(input("Chunk size rows [50000]: ") or "50000")

        Path(output_dir).mkdir(exist_ok=True, parents=True)

        rank_and_filter_chunks_optimized(
            chunks_tsv=chunks_tsv,
            output_dir=Path(output_dir),
            original_tsv=original_tsv,
            upper_percentile=up,
            lower_percentile=low,
            model_name=model,
            max_workers=workers,
            chunk_size=chunk_sz,
        )
