"""
Optimized ranking module with memory efficiency and caching
"""
import pandas as pd
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


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
                                    upper_percentile: int = 80, lower_percentile: int = 20,
                                    model_name: str = "thenlper/gte-base", max_workers: int = None) -> str:
    """Optimized version of rank_and_filter_chunks with memory efficiency and parallel processing
    
    Args:
        chunks_tsv: Path to input TSV file
        output_dir: Output directory
        upper_percentile: Upper percentile for positive samples
        lower_percentile: Lower percentile for negative samples
        model_name: Embedding model name
        max_workers: Number of parallel workers (None = sequential processing)
    """
    
    if not Path(chunks_tsv).exists():
        return ""
    
    print(f"Loading data from {chunks_tsv}...")
    df = pd.read_csv(chunks_tsv, sep='\t', on_bad_lines='warn', engine='python')
    df.columns = df.columns.str.strip()
    
    required_cols = {'query_id', 'query_text', 'chunk_text', 'chunk_id'}
    if not required_cols.issubset(set(df.columns)):
        print(f"Missing required columns. Found: {df.columns.tolist()}")
        return ""
    
    total_queries = df['query_id'].nunique()
    print(f"Processing {total_queries} unique queries...")
    
    # Choose processing method based on max_workers
    if max_workers and max_workers > 1:
        print(f"Using parallel processing with {max_workers} workers (dedicated models)")
        kept_rows = _process_queries_parallel(df, model_name, upper_percentile, lower_percentile, max_workers)
    else:
        print("Using sequential processing (single model)")
        kept_rows = _process_queries_sequential(df, model_name, upper_percentile, lower_percentile)
    
    if not kept_rows:
        print("No data to save after filtering")
        return ""
    
    # Combine results
    print("Combining and saving results...")
    final_df = pd.concat(kept_rows, ignore_index=True)
    
    # Save main file
    save_path = output_dir / f"{Path(chunks_tsv).stem}_rrf_filtered.tsv"
    final_df.to_csv(save_path, sep='\t', index=False)
    
    # Save 3-column version
    try:
        simple_df = final_df[['query_text', 'chunk_text', 'label']].copy()
        simple_df.columns = ['query', 'passage', 'label']
        simple_path = output_dir / f"{Path(chunks_tsv).stem}_rrf_filtered_3col.tsv"
        simple_df.to_csv(simple_path, sep='\t', index=False)
    except Exception as e:
        print(f"Warning: Could not create 3-column file: {e}")
    
    print(f"Ranking complete. Saved {len(final_df)} filtered chunks to {save_path}")
    
    # Force cleanup
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
