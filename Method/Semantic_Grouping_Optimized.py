import pandas as pd
from dotenv import load_dotenv
from Tool.Sentence_Segmenter import extract_sentences_spacy, count_tokens_spacy
from Tool.Sentence_Embedding import sentence_embedding as embed_text_list 
from Tool.OIE import extract_relations_from_paragraph 
import os 
import json 
from typing import List, Tuple, Dict, Union, Optional 
import gc 
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np 
import hashlib 
from datetime import datetime
from pathlib import Path

load_dotenv()

def _count_tokens_accurate(text: str) -> int:
    """Token counting using spaCy tokenizer"""
    return count_tokens_spacy(text)

def process_sentence_embedding_in_batches(sentences, model_name, batch_size=32, device=None, silent=True):
    """
    Embeds a list of sentences in batches using a specified sentence transformer model.
    Optimized for better memory management and error handling.
    """
    if not sentences:
        if not silent:
            print("No sentences provided to embed.")
        return np.array([])

    # Dynamic batch size optimization based on sentence count
    if len(sentences) <= 10:
        batch_size = min(batch_size, len(sentences))  # Process all at once for tiny datasets
    elif len(sentences) < 50:
        batch_size = min(batch_size, 8)  # Very small batches for small datasets
    elif len(sentences) > 200:
        batch_size = min(batch_size * 2, 64)  # Larger batches for big datasets

    all_embeddings = []
    if not silent:
        print(f"Embedding {len(sentences)} sentences in batches of {batch_size}...")

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(sentences) + batch_size - 1) // batch_size
        
        if not silent and (batch_num % max(1, total_batches // 10) == 0):  # Progress every 10%
            progress = (batch_num / total_batches) * 100
            print(f"  Embedding progress: {progress:.1f}% ({batch_num}/{total_batches})")
        
        try:
            device_pref = str(device) if device is not None else "cpu"
            batch_embeddings = embed_text_list(batch, model_name=model_name, batch_size=len(batch), device_preference=device_pref)
            if batch_embeddings is not None and len(batch_embeddings) > 0:
                all_embeddings.append(batch_embeddings)
            else:
                if not silent:
                    print(f"Warning: Batch {batch_num} returned no embeddings.")
        except Exception as e:
            if not silent:
                print(f"Error during batched embedding batch {batch_num}: {e}")

    if not all_embeddings:
        if not silent:
            print("No embeddings were generated successfully.")
        return np.array([])

    try:
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings
    except ValueError as e:
        if not silent:
            print(f"Error concatenating batch embeddings: {e}")
        return np.array([])

def helper_create_semantic_matrix(sentences: List[str], model_name: str, batch_size_embed: int = 32, device: Optional[str] = None, silent: bool = True) -> Optional[np.ndarray]:
    """Create semantic similarity matrix between sentences using batched embedding."""
    if len(sentences) < 2:
        return None

    # Smart batch size for small documents
    num_sentences = len(sentences)
    if num_sentences <= 10:
        # For very small documents, process all sentences in one batch
        optimal_batch_size = num_sentences
    elif num_sentences <= 30:
        # For small documents, use small batches
        optimal_batch_size = min(8, num_sentences)
    else:
        # For larger documents, use configured batch size
        optimal_batch_size = min(max(batch_size_embed, 16), 64)
    
    if not silent and num_sentences <= 10:
        print(f"Small document ({num_sentences} sentences): Processing in single batch")
    
    embeddings = process_sentence_embedding_in_batches(
        sentences, 
        model_name=model_name, 
        batch_size=optimal_batch_size, 
        device=device, 
        silent=silent
    )

    if embeddings is None or embeddings.shape[0] != len(sentences):
        if not silent:
            print("Error: Embedding failed or mismatch in number of embeddings.")
        if 'embeddings' in locals() and embeddings is not None: 
            del embeddings
        gc.collect()
        return None

    try:
        # Use float32 for faster computation and less memory
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        sim_matrix = cosine_similarity(embeddings)
        del embeddings
        gc.collect()
        return sim_matrix
    except Exception as e:
        if not silent:
            print(f"Error calculating similarity matrix: {e}")
        if 'embeddings' in locals() and embeddings is not None: 
            del embeddings
        gc.collect()
        return None

def analyze_similarity_distribution(sim_matrix):
    """Analyze similarity distribution in matrix, excluding diagonal values."""
    if not isinstance(sim_matrix, np.ndarray) or sim_matrix.ndim != 2 or sim_matrix.shape[0] < 2:
        return None

    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[upper_triangle_indices]

    epsilon = 1e-5
    filtered_similarities = similarities[similarities < (1.0 - epsilon)]

    if filtered_similarities.size == 0:
        if similarities.size > 0:
            original_max = np.max(similarities)
            return {
                'min': original_max, 'max': original_max, 'mean': original_max, 'std': 0.0,
                **{f'p{p}': original_max for p in [10, 25, 50, 75, 80, 85, 90, 95]}
            }
        return None

    percentiles = {
        f'p{p}': np.percentile(filtered_similarities, p) for p in [10, 25, 50, 75, 80, 85, 90, 95]
    }
    stats = {
        'min': np.min(filtered_similarities),
        'max': np.max(filtered_similarities),
        'mean': np.mean(filtered_similarities),
        'std': np.std(filtered_similarities),
        **percentiles
    }

    return stats

def helper_find_best_similarity_pairs(sim_matrix, ungrouped_indices, sentences, threshold, max_tokens, silent):
    """Find pairs of sentences that meet similarity and token constraints - optimized version"""
    if len(ungrouped_indices) < 2:
        return []
        
    valid_pairs = []
    ungrouped_array = np.array(ungrouped_indices)
    
    # Vectorized similarity extraction for all pairs
    indices_mesh = np.meshgrid(ungrouped_array, ungrouped_array, indexing='ij')
    i_indices, j_indices = indices_mesh[0].flatten(), indices_mesh[1].flatten()
    
    # Filter to upper triangle only (avoid duplicates and self-pairs)
    mask = i_indices < j_indices
    i_filtered, j_filtered = i_indices[mask], j_indices[mask]
    
    if len(i_filtered) == 0:
        return []
    
    # Get similarities for all valid pairs at once
    similarities = sim_matrix[i_filtered, j_filtered]
    
    # Filter by threshold
    threshold_mask = similarities >= threshold
    if not np.any(threshold_mask):
        return []
    
    valid_i = i_filtered[threshold_mask]
    valid_j = j_filtered[threshold_mask]
    valid_sims = similarities[threshold_mask]
    
    # Check token constraints for remaining pairs
    for idx, (i, j, sim) in enumerate(zip(valid_i, valid_j, valid_sims)):
        combined_text = sentences[i] + " " + sentences[j]
        combined_tokens = _count_tokens_accurate(combined_text)
        
        if combined_tokens <= max_tokens:
            valid_pairs.append(((i, j), sim))
            
        # Limit checking to top candidates for performance
        if len(valid_pairs) >= 10:  # Only check top 10 by similarity
            break

    # Sort by similarity score descending
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    return valid_pairs

def helper_expand_group_with_constraints(sim_matrix, sentences, current_group, ungrouped_indices, 
                                threshold, max_tokens, silent):
    """Expand a group by adding similar sentences while respecting token constraints"""
    added_in_iteration = True
    max_expansion_iterations = len(ungrouped_indices) + 1
    expansion_iteration = 0
    
    while added_in_iteration and expansion_iteration < max_expansion_iterations:
        expansion_iteration += 1
        added_in_iteration = False
        candidates_to_add = []
        
        for ungrouped_idx in list(ungrouped_indices):
            max_similarity = -1
            for group_member_idx in current_group:
                similarity = sim_matrix[ungrouped_idx, group_member_idx]
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= threshold:
                test_group_sentences = [sentences[idx] for idx in current_group + [ungrouped_idx]]
                test_text = " ".join(test_group_sentences)
                test_tokens = _count_tokens_accurate(test_text)
                
                if test_tokens <= max_tokens:
                    candidates_to_add.append((ungrouped_idx, max_similarity))

        if candidates_to_add:
            candidates_to_add.sort(key=lambda x: x[1], reverse=True)
            for candidate_idx, similarity in candidates_to_add:
                if candidate_idx in ungrouped_indices:
                    current_group.append(candidate_idx)
                    ungrouped_indices.remove(candidate_idx)
                    added_in_iteration = True
                    if not silent:
                        print(f"    Added sentence {candidate_idx} (similarity: {similarity:.3f})")

    return current_group

def helper_handle_remaining_sentences(remaining_indices, sentences, min_tokens, max_tokens, silent):
    """Handle sentences that couldn't be grouped, trying to merge small ones"""
    if not remaining_indices:
        return []
    
    groups = []
    current_merge_group = []
    current_merge_tokens = 0
    
    if not silent:
        print(f"Handling {len(remaining_indices)} remaining sentences...")
    
    for idx in remaining_indices:
        sentence_tokens = _count_tokens_accurate(sentences[idx])
        
        if sentence_tokens >= min_tokens:
            if sentence_tokens <= max_tokens:
                groups.append([idx])
                if not silent:
                    print(f"  Individual sentence {idx}: {sentence_tokens} tokens ✓")
        else:
            test_tokens = current_merge_tokens + sentence_tokens
            if test_tokens <= max_tokens:
                current_merge_group.append(idx)
                current_merge_tokens = test_tokens
            else:
                if current_merge_tokens >= min_tokens and current_merge_group:
                    groups.append(current_merge_group)
                    if not silent:
                        print(f"  Merged group: {len(current_merge_group)} sentences, {current_merge_tokens} tokens ✓")
                
                current_merge_group = [idx]
                current_merge_tokens = sentence_tokens
    
    if current_merge_group:
        if current_merge_tokens >= min_tokens:
            groups.append(current_merge_group)
            if not silent:
                print(f"  Final merged group: {len(current_merge_group)} sentences, {current_merge_tokens} tokens ✓")
    
    return groups

def process_semantic_grouping_optimized(sim_matrix, sentences, initial_threshold, decay_factor, min_threshold, 
                                        min_chunk_len, max_chunk_len, silent=True):
    """Optimized semantic grouping algorithm with early termination and fast path for small documents"""
    num_sentences = sim_matrix.shape[0]
    
    # Fast path for very small documents (≤6 sentences)
    if num_sentences <= 6:
        if not silent:
            print(f"Fast path for small document ({num_sentences} sentences)")
        
        # For small documents, try to group all sentences if they meet token constraints
        all_text = " ".join(sentences)
        all_tokens = _count_tokens_accurate(all_text)
        
        if all_tokens <= max_chunk_len:
            if not silent:
                print(f"  Single group: {num_sentences} sentences, {all_tokens} tokens ✓")
            return [list(range(num_sentences))]
        
        # If too big, use simplified pairing approach
        groups = []
        remaining = list(range(num_sentences))
        
        while remaining:
            current_group = [remaining.pop(0)]
            current_text = sentences[current_group[0]]
            current_tokens = _count_tokens_accurate(current_text)
            
            # Try to add more sentences to current group
            i = 0
            while i < len(remaining):
                test_text = current_text + " " + sentences[remaining[i]]
                test_tokens = _count_tokens_accurate(test_text)
                
                if test_tokens <= max_chunk_len:
                    current_group.append(remaining.pop(i))
                    current_text = test_text
                    current_tokens = test_tokens
                else:
                    i += 1
            
            if current_tokens >= min_chunk_len:
                groups.append(current_group)
                if not silent:
                    print(f"  Small doc group: {len(current_group)} sentences, {current_tokens} tokens ✓")
        
        return groups
    
    # Standard processing for larger documents
    ungrouped_indices = set(range(num_sentences))
    groups = []
    current_threshold = initial_threshold
    iteration = 0
    max_iterations = num_sentences * 2
    consecutive_empty_iterations = 0  # Track empty iterations for early exit

    if not silent:
        print(f"Starting semantic grouping with {num_sentences} sentences")
        print(f"Token constraints: {min_chunk_len}-{max_chunk_len}")
        print(f"Thresholds: initial={initial_threshold:.3f}, decay={decay_factor}, min={min_threshold:.3f}")

    while len(ungrouped_indices) > 0 and iteration < max_iterations:
        iteration += 1
        
        best_pairs = helper_find_best_similarity_pairs(
            sim_matrix, list(ungrouped_indices), sentences, 
            current_threshold, max_chunk_len, silent
        )
        
        if not best_pairs:
            consecutive_empty_iterations += 1
            new_threshold = max(min_threshold, current_threshold * decay_factor)
            if new_threshold == current_threshold or consecutive_empty_iterations >= 3:
                # Early termination if threshold can't decay or too many empty iterations
                if not silent:
                    print(f"  Early termination: threshold={current_threshold:.4f}, empty_iterations={consecutive_empty_iterations}")
                break
            current_threshold = new_threshold
            if not silent:
                print(f"  No valid pairs found. Decaying threshold to {current_threshold:.4f}")
            continue
        
        # Reset empty iteration counter when we find pairs
        consecutive_empty_iterations = 0

        best_pair, similarity_score = best_pairs[0]
        idx1, idx2 = best_pair
        
        current_group = [idx1, idx2]
        ungrouped_indices.discard(idx1)
        ungrouped_indices.discard(idx2)
        
        current_group = helper_expand_group_with_constraints(
            sim_matrix, sentences, current_group, list(ungrouped_indices),
            current_threshold, max_chunk_len, silent
        )
        
        for idx in current_group:
            ungrouped_indices.discard(idx)
        
        group_text = " ".join([sentences[idx] for idx in current_group])
        group_tokens = _count_tokens_accurate(group_text)
        
        if group_tokens >= min_chunk_len:
            groups.append(sorted(current_group))
            if not silent:
                print(f"  Group {len(groups)}: {len(current_group)} sentences, {group_tokens} tokens ✓")
        else:
            if not silent:
                print(f"  Group candidate: {group_tokens} tokens < {min_chunk_len}, discarding")
            ungrouped_indices.update(current_group)

        current_threshold = max(min_threshold, initial_threshold * (decay_factor ** len(groups)))

    if ungrouped_indices:
        remaining_groups = helper_handle_remaining_sentences(
            list(ungrouped_indices), sentences, min_chunk_len, max_chunk_len, silent
        )
        groups.extend(remaining_groups)

    if not silent:
        print(f"Semantic grouping completed: {len(groups)} groups formed in {iteration} iterations")

    return groups

def format_oie_triples_to_string(triples_list: List[Dict[str, str]], max_triples: Optional[int] = None) -> Optional[str]:
    """Format OIE triples into a readable string format"""
    if not triples_list:
        return None
    
    formatted_sentences = []
    triples_to_process = triples_list[:max_triples] if max_triples is not None else triples_list

    for triple in triples_to_process:
        s = str(triple.get('subject', '')).replace('\t', ' ').replace('\n', ' ').strip()
        r = str(triple.get('relation', '')).replace('\t', ' ').replace('\n', ' ').strip()
        o = str(triple.get('object', '')).replace('\t', ' ').replace('\n', ' ').strip()
        
        if s and r and o:
            formatted_sentences.append(f"{s} {r} {o}.")
    
    if not formatted_sentences:
        return None

    return " ".join(formatted_sentences).strip()

def helper_force_split_too_large_chunk(text: str, max_tokens: int) -> List[str]:
    """Splits a chunk of text that is over max_tokens into smaller chunks."""
    sentences = extract_sentences_spacy(text)
    if not sentences:
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_len = _count_tokens_accurate(sentence)

        if sentence_len > max_tokens:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0
            
            words = sentence.split()
            sub_chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
            chunks.extend(sub_chunks)
            continue

        if current_chunk_sentences and current_tokens + sentence_len > max_tokens:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_tokens = sentence_len
        else:
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_len
    
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return [chunk for chunk in chunks if chunk]

def process_oie_extraction(chunk_text: str, max_triples: Optional[int] = None, silent: bool = True) -> Optional[str]:
    """Extract OIE relations for a chunk and format them"""
    if not chunk_text or not chunk_text.strip():
        return None
    
    try:
        if not silent:
            print(f"    Extracting OIE for chunk ({_count_tokens_accurate(chunk_text)} tokens)...")
        
        relations = extract_relations_from_paragraph(chunk_text)
        
        if relations:
            oie_string = format_oie_triples_to_string(relations, max_triples=max_triples)
            if not silent and oie_string:
                print(f"      Found {len(relations)} OIE relations")
            return oie_string
        else:
            if not silent:
                print(f"      No OIE relations found")
            return None
            
    except Exception as e:
        if not silent:
            print(f"      Error during OIE extraction: {e}")
        return None

def semantic_grouping_main(
    passage_text: str,
    doc_id: str,
    embedding_model: str,  
    initial_threshold: Union[str, float] = "auto",
    decay_factor: float = 0.85,
    min_threshold: Union[str, float] = "auto",
    min_chunk_len_tokens: int = 0,  # Simplified: no minimum constraint      
    max_chunk_len_tokens: int = 200,     
    include_oie: bool = False,
    save_raw_oie: bool = False,
    output_dir: str = "./output",
    initial_percentile: str = "85",  
    min_percentile: str = "25",
    embedding_batch_size: int = 16,
    silent: bool = False,
    **kwargs
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Enhanced semantic grouping with improved algorithm and OIE support.
    
    Returns:
        List of tuples: (chunk_id, chunk_text_with_oie, oie_string_only)
    """

    if not silent:
        print(f"    Processing passage {doc_id} with Enhanced Semantic Grouping")
        print(f"   Model: {embedding_model}")
        print(f"   Token range: {min_chunk_len_tokens}-{max_chunk_len_tokens}")
        print(f"   OIE enabled: {include_oie}")

    # Extract sentences
    sentences = extract_sentences_spacy(passage_text)
    if not sentences:
        if not silent:
            print(f"   No sentences found in passage {doc_id}")
        
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = process_oie_extraction(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        return [(f"{doc_id}_single_empty", final_text, oie_string)]

    # Check if passage is small enough to keep as single chunk
    total_tokens = _count_tokens_accurate(passage_text)
    if total_tokens <= max_chunk_len_tokens:
        if not silent:
            print(f"   Passage fits in single chunk ({total_tokens} tokens)")
        
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = process_oie_extraction(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        return [(f"{doc_id}_single_complete", final_text, oie_string)]

    # Create similarity matrix
    if not silent:
        print(f"   Creating similarity matrix for {len(sentences)} sentences...")
    
    sim_matrix = helper_create_semantic_matrix(
        sentences, 
        model_name=embedding_model, 
        batch_size_embed=embedding_batch_size, 
        device=kwargs.get('device'), 
        silent=silent
    )
    
    if sim_matrix is None:
        if not silent:
            print(f"   Failed to create similarity matrix")
        
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = process_oie_extraction(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        return [(f"{doc_id}_matrix_fail", final_text, oie_string)]

    # Resolve auto thresholds
    actual_initial_threshold = initial_threshold
    actual_min_threshold = min_threshold
    
    if initial_threshold == 'auto' or min_threshold == 'auto':
        if not silent:
            print(f"   Analyzing similarity distribution for auto thresholds...")
        
        dist_stats = analyze_similarity_distribution(sim_matrix)
        if dist_stats:
            if initial_threshold == 'auto':
                percentile_key = f'p{initial_percentile}'
                actual_initial_threshold = dist_stats.get(percentile_key, 0.75)
                if not silent:
                    print(f"   Auto initial_threshold ({initial_percentile}th percentile): {actual_initial_threshold:.4f}")
            
            if min_threshold == 'auto':
                percentile_key = f'p{min_percentile}'
                actual_min_threshold = dist_stats.get(percentile_key, 0.5)
                if not silent:
                    print(f"   Auto min_threshold ({min_percentile}th percentile): {actual_min_threshold:.4f}")
            
            try:
                init_val = float(actual_initial_threshold)
                min_val  = float(actual_min_threshold)
            except (TypeError, ValueError):
                init_val, min_val = 0.75, 0.5

            if init_val < min_val:
                min_val = init_val * 0.7

            actual_initial_threshold = init_val
            actual_min_threshold = min_val
        else:
            if not silent:
                print(f"   Could not analyze distribution, using defaults")
            actual_initial_threshold = 0.75 if initial_threshold == 'auto' else float(initial_threshold)
            actual_min_threshold = 0.5 if min_threshold == 'auto' else float(min_threshold)
    else:
        actual_initial_threshold = float(initial_threshold)
        actual_min_threshold = float(min_threshold)

    # Perform semantic grouping
    if not silent:
        print(f"   Grouping with thresholds: {actual_initial_threshold:.4f} → {actual_min_threshold:.4f}")
    
    group_indices = process_semantic_grouping_optimized(
        sim_matrix, sentences, actual_initial_threshold, decay_factor, 
        actual_min_threshold, min_chunk_len_tokens, max_chunk_len_tokens, silent
    )
    
    del sim_matrix
    gc.collect()

    if not group_indices:
        if not silent:
            print(f"   No groups formed, returning original passage")
        
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = process_oie_extraction(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        return [(f"{doc_id}_no_groups", final_text, oie_string)]

    # Create final chunks with OIE
    if not silent:
        print(f"   Creating {len(group_indices)} chunks with OIE processing...")
    
    final_chunks = []
    all_raw_oie_data: List[Dict] = []
    
    for i, group_sentence_indices in enumerate(group_indices):
        valid_indices = [idx for idx in group_sentence_indices if 0 <= idx < len(sentences)]
        if not valid_indices:
            continue
        
        chunk_sentences = [sentences[idx] for idx in sorted(valid_indices)]
        chunk_text = " ".join(chunk_sentences).strip()
        
        if not chunk_text:
            continue
        
        chunk_hash = hashlib.sha1(chunk_text.encode('utf-8', errors='ignore')).hexdigest()[:8]
        chunk_id = f"{doc_id}_group{i}_hash{chunk_hash}"
        
        oie_string = None
        final_chunk_text = chunk_text
        raw_oie_relations = None

        if include_oie and chunk_text.strip():
            try:
                raw_oie_relations = extract_relations_from_paragraph(chunk_text, silent=True)
                if raw_oie_relations:
                    oie_string = format_oie_triples_to_string(raw_oie_relations)
                    if oie_string:
                        final_chunk_text = f"{chunk_text} {oie_string}"

                    if save_raw_oie:
                        all_raw_oie_data.append({
                            "chunk_id": chunk_id,
                            "relations": raw_oie_relations,
                            "relation_count": len(raw_oie_relations),
                            "chunk_text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                        })
            except Exception as e_oie:
                if not silent:
                    print(f"      Error during OIE extraction: {e_oie}")
                oie_string = None
        
        base_tokens = _count_tokens_accurate(chunk_text)
        if base_tokens >= min_chunk_len_tokens:
            final_chunks.append((chunk_id, final_chunk_text, oie_string))
            if not silent:
                final_tokens = _count_tokens_accurate(final_chunk_text)
                print(f"   ✓ Chunk {i}: {base_tokens} base tokens, {final_tokens} final tokens")
        else:
            if not silent:
                print(f"   ✗ Chunk {i}: {base_tokens} base tokens < {min_chunk_len_tokens} (discarded)")

    # Handle case where no valid chunks were created
    if not final_chunks:
        if not silent:
            print(f"   No valid chunks after filtering, returning original passage")
        
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = process_oie_extraction(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        final_chunks = [(f"{doc_id}_filter_fail", final_text, oie_string)]

    # Force-split any chunks that are still too large
    if not silent:
        print(f"   Ensuring all chunks respect max token limit ({max_chunk_len_tokens})...")

    force_split_chunks = []
    for chunk_id, chunk_text_w_oie, oie_string_from_chunker in final_chunks:
        base_text = chunk_text_w_oie
        if oie_string_from_chunker and base_text.endswith(oie_string_from_chunker):
             base_text = base_text[:-len(oie_string_from_chunker)].strip()
        
        base_token_count = _count_tokens_accurate(base_text)

        if base_token_count > max_chunk_len_tokens:
            if not silent:
                print(f"     Chunk {chunk_id} ({base_token_count} tokens) is too large. Force-splitting...")
            
            sub_chunks_text = helper_force_split_too_large_chunk(base_text, max_chunk_len_tokens)
            
            for i, sub_chunk_text_item in enumerate(sub_chunks_text):
                sub_chunk_id = f"{chunk_id}_fs{i}"
                
                sub_oie_string = None
                final_sub_chunk_text = sub_chunk_text_item
                if include_oie:
                    sub_oie_string = process_oie_extraction(sub_chunk_text_item, silent=silent)
                    if sub_oie_string:
                        final_sub_chunk_text = f"{sub_chunk_text_item} {sub_oie_string}"
                
                force_split_chunks.append((sub_chunk_id, final_sub_chunk_text, sub_oie_string))
        else:
            force_split_chunks.append((chunk_id, chunk_text_w_oie, oie_string_from_chunker))

    final_chunks = force_split_chunks

    # Save raw OIE if needed
    if save_raw_oie and all_raw_oie_data:
        try:
            raw_oie_filepath = helper_save_raw_oie_data(all_raw_oie_data, doc_id, output_dir, "semantic_grouping")
            if raw_oie_filepath and not silent:
                print(f"📄 Raw OIE data saved to: {raw_oie_filepath}")
                total_relations = sum(entry['relation_count'] for entry in all_raw_oie_data)
                print(f"   Total chunks with OIE: {len(all_raw_oie_data)}")
                print(f"   Total relations extracted: {total_relations}")
        except Exception as e:
            if not silent:
                print(f"Warning: Failed to save raw OIE data: {e}")

    if not silent:
        token_counts = [_count_tokens_accurate(chunk[1]) for chunk in final_chunks]
        if token_counts:
            print(f"  Created {len(final_chunks)} final chunks after force-splitting.")
            print(f"  Final token distribution: {min(token_counts)}-{max(token_counts)} (avg: {sum(token_counts)/len(token_counts):.1f})")
        else:
            print("   No chunks were created after processing.")

    return final_chunks

def helper_save_raw_oie_data(oie_data: List[Dict], chunk_id: str, output_dir: str, method_name: str = "semantic_grouping") -> Optional[str]:
    """Save raw OIE data to JSON file for analysis"""
    try:
        raw_oie_dir = Path(output_dir) / "raw_oie_data"
        raw_oie_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{method_name}_raw_oie_{timestamp}.json"
        filepath = raw_oie_dir / filename

        oie_entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
            "chunk_id": chunk_id,
            "raw_oie_relations": oie_data,
            "total_relations": sum(e['relation_count'] for e in oie_data)
        }

        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_data.append(oie_entry)
        else:
            existing_data = [oie_entry]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        return str(filepath)
    except Exception as e:
        print(f"Warning: Could not save raw OIE data: {e}")
        return None
