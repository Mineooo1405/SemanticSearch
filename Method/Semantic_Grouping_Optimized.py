import pandas as pd
from dotenv import load_dotenv
from Tool.Sentence_Segmenter import extract_sentences_spacy
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

def _normalize_device(device) -> str:
    """Normalize device to 'cuda' or 'cpu'"""
    if device is None:
        return "cuda"
    device_str = str(device).lower()
    return "cuda" if device_str == "cuda" else "cpu"

def _optimize_gpu_batch_size(sentences, base_batch_size, device):
    """Optimize batch size based on sentence complexity and GPU memory"""
    device = _normalize_device(device)
    if device != "cuda":
        return base_batch_size
    
    # Estimate complexity based on average sentence length
    avg_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 100
    
    # Aggressive sizing for RTX 5080 with large VRAM
    if avg_length < 50:  # Short sentences
        return min(256, len(sentences))
    elif avg_length < 100:  # Medium sentences  
        return min(128, len(sentences))
    elif avg_length < 200:  # Long sentences
        return min(64, len(sentences))
    else:  # Very long sentences
        return min(32, len(sentences))

def process_sentence_embedding_in_batches(sentences, model_name, batch_size=32, device="cuda", silent=True):
    """
    Embeds a list of sentences in batches using a specified sentence transformer model.
    Optimized for better memory management and GPU utilization.
    """
    if not sentences:
        if not silent:
            print("No sentences provided to embed.")
        return np.array([])
    
    device = _normalize_device(device)
    
    # Log device and optimization info
    if not silent:
        print(f"Embedding Setup: Device={device}, Model={model_name}")
        print(f"Input: {len(sentences)} sentences, Base batch size: {batch_size}")

    # Optimize batch size for GPU utilization
    if device == "cuda":
        optimized_batch_size = _optimize_gpu_batch_size(sentences, batch_size, device)
        if not silent:
            print(f"   GPU Optimization: {batch_size} → {optimized_batch_size} (RTX 5080 optimized)")
        batch_size = optimized_batch_size
    else:
        # Conservative CPU batching
        if len(sentences) <= 10:
            batch_size = min(batch_size, len(sentences))
        elif len(sentences) < 50:
            batch_size = min(batch_size, 8)
        elif len(sentences) > 200:
            batch_size = min(batch_size * 2, 64)

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
            # For GPU, use larger internal batch size for better utilization
            internal_batch_size = len(batch) if device == "cuda" else min(len(batch), 16)
            batch_embeddings = embed_text_list(
                batch, 
                model_name=model_name, 
                batch_size=internal_batch_size, 
                device_preference=device
            )
            if batch_embeddings is not None and len(batch_embeddings) > 0:
                all_embeddings.append(batch_embeddings)
            else:
                if not silent:
                    print(f"Warning: Batch {batch_num} returned no embeddings.")
        except Exception as e:
            if not silent:
                print(f"Error during batched embedding batch {batch_num}: {e}")
                print(f"Device being used: {device}, Batch size: {len(batch)}")

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

def helper_create_semantic_matrix(sentences: List[str], model_name: str, batch_size_embed: int = 32, device: Optional[str] = "cuda", silent: bool = True) -> Optional[np.ndarray]:
    """Create semantic similarity matrix between sentences using batched embedding."""
    if len(sentences) < 2:
        return None

    device = _normalize_device(device)
    
    # GPU-optimized batch sizing using the new optimization function
    num_sentences = len(sentences)
    if device == "cuda":
        optimal_batch_size = _optimize_gpu_batch_size(sentences, batch_size_embed, device)
    else:
        # Conservative CPU batching
        if num_sentences <= 10:
            optimal_batch_size = num_sentences
        elif num_sentences <= 30:
            optimal_batch_size = min(8, num_sentences)
        else:
            optimal_batch_size = min(max(batch_size_embed, 16), 64)
    
    if not silent:
        device_info = f"GPU ({device})" if device == "cuda" else f"CPU ({device})"
        print(f"Matrix Creation: {num_sentences} sentences on {device_info}")
        print(f"Optimal batch size: {optimal_batch_size}")
    
    import time
    start_time = time.time()
    
    embeddings = process_sentence_embedding_in_batches(
        sentences, 
        model_name=model_name, 
        batch_size=optimal_batch_size, 
        device=device, 
        silent=silent
    )
    
    embedding_time = time.time() - start_time
    if not silent:
        print(f"   Embedding completed in {embedding_time:.2f}s ({len(sentences)/embedding_time:.1f} sentences/sec)")

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

def helper_find_best_similarity_pairs(sim_matrix, ungrouped_indices, sentences, threshold, silent):
    """Find pairs of sentences that meet similarity constraints - optimized version"""
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
    
    # Create valid pairs without token checking
    for i, j, sim in zip(valid_i, valid_j, valid_sims):
        valid_pairs.append(((i, j), sim))
        
        # Limit checking to top candidates for performance
        if len(valid_pairs) >= 20:  # Check more pairs since no token constraint
            break

    # Sort by similarity score descending
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    return valid_pairs

def helper_expand_group_with_constraints(sim_matrix, sentences, current_group, ungrouped_indices, 
                                threshold, silent):
    """Expand a group by adding similar sentences"""
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

def helper_handle_remaining_sentences(remaining_indices, sentences, silent):
    """Handle sentences that couldn't be grouped"""
    if not remaining_indices:
        return []
    
    groups = []
    
    if not silent:
        print(f"Handling {len(remaining_indices)} remaining sentences...")
    
    # Simply put each remaining sentence in its own group
    for idx in remaining_indices:
        groups.append([idx])
        if not silent:
            print(f"  Individual sentence {idx} ✓")
    
    return groups

def process_semantic_grouping_with_threshold_ranges(sim_matrix, sentences, initial_threshold, decay_factor, min_threshold, silent=True):
    """
    New semantic grouping algorithm using threshold ranges.
    
    Process chunks in descending threshold ranges:
    - Range 1: [initial_threshold, initial_threshold*decay_factor] 
    - Range 2: [initial_threshold*decay_factor, initial_threshold*decay_factor^2]
    - Range 3: [initial_threshold*decay_factor^2, initial_threshold*decay_factor^3]
    - ... until min_threshold or no sentences left
    """
    num_sentences = sim_matrix.shape[0]
    
    # Fast path for very small documents (≤6 sentences)
    if num_sentences <= 6:
        if not silent:
            print(f"Fast path for small document ({num_sentences} sentences)")
            print(f"  Single group: {num_sentences} sentences ✓")
        return [list(range(num_sentences))]
    
    ungrouped_indices = set(range(num_sentences))
    groups = []
    
    # Calculate threshold ranges
    threshold_ranges = []
    current_upper = initial_threshold
    range_index = 0
    
    while current_upper >= min_threshold and range_index < 20:  # Max 20 ranges to prevent infinite loop
        current_lower = max(min_threshold, current_upper * decay_factor)
        threshold_ranges.append((current_upper, current_lower))
        current_upper = current_lower
        range_index += 1
        
        # Stop if we've reached min_threshold
        if current_lower <= min_threshold:
            break
    
    if not silent:
        print(f"Starting threshold range grouping with {num_sentences} sentences")
        print(f"Threshold ranges: {len(threshold_ranges)} ranges from {initial_threshold:.3f} to {min_threshold:.3f}")
        for i, (upper, lower) in enumerate(threshold_ranges):
            print(f"  Range {i+1}: [{upper:.3f}, {lower:.3f})")
    
    # Process each threshold range
    for range_idx, (upper_threshold, lower_threshold) in enumerate(threshold_ranges):
        if len(ungrouped_indices) == 0:
            break  # Early exit if no sentences left
            
        if not silent:
            print(f"\n--- Processing Range {range_idx + 1}: [{upper_threshold:.3f}, {lower_threshold:.3f}) ---")
        
        range_groups = process_range_grouping(
            sim_matrix, 
            sentences, 
            list(ungrouped_indices),
            upper_threshold, 
            lower_threshold, 
            silent
        )
        
        # Add valid groups and remove grouped sentences
        for group in range_groups:
            if group:  # Only add non-empty groups
                groups.append(sorted(group))
                for idx in group:
                    ungrouped_indices.discard(idx)
                    
                if not silent:
                    print(f"  ✓ Group {len(groups)}: {len(group)} sentences (range {range_idx + 1})")
    
    # Handle remaining ungrouped sentences
    if ungrouped_indices:
        if not silent:
            print(f"\n--- Handling {len(ungrouped_indices)} remaining sentences ---")
        
        remaining_groups = helper_handle_remaining_sentences(
            list(ungrouped_indices), sentences, silent
        )
        groups.extend(remaining_groups)
    
    if not silent:
        print(f"\nThreshold range grouping completed: {len(groups)} groups formed")
        
    return groups

def process_range_grouping(sim_matrix, sentences, available_indices, upper_threshold, lower_threshold, silent=True):
    """
    Process grouping within a specific threshold range.
    Find all similarity pairs within [lower_threshold, upper_threshold) and group them.
    """
    if len(available_indices) < 2:
        return []
    
    # Find all valid pairs within the threshold range
    valid_pairs = []
    for i in range(len(available_indices)):
        for j in range(i + 1, len(available_indices)):
            idx1, idx2 = available_indices[i], available_indices[j]
            similarity = sim_matrix[idx1, idx2]
            
            # Check if similarity is within the range [lower_threshold, upper_threshold)
            if lower_threshold <= similarity < upper_threshold:
                valid_pairs.append(((idx1, idx2), similarity))
    
    if not valid_pairs:
        if not silent:
            print(f"    No pairs found in range [{upper_threshold:.3f}, {lower_threshold:.3f})")
        return []
    
    # Sort pairs by similarity (highest first)
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    
    if not silent:
        print(f"    Found {len(valid_pairs)} valid pairs in range")
    
    # Group formation using Union-Find like approach
    groups = []
    used_indices = set()
    
    # Process pairs from highest to lowest similarity
    for (idx1, idx2), similarity in valid_pairs:
        if idx1 in used_indices or idx2 in used_indices:
            continue  # Skip if either sentence is already grouped
            
        # Start new group with this pair
        current_group = [idx1, idx2]
        used_indices.add(idx1)
        used_indices.add(idx2)
        
        # Try to expand the group by finding more sentences that fit
        remaining_available = [idx for idx in available_indices if idx not in used_indices]
        
        if remaining_available:
            expanded_group = expand_group_in_range(
                sim_matrix, 
                current_group, 
                remaining_available, 
                upper_threshold, 
                lower_threshold,
                silent
            )
            
            # Update used indices
            for idx in expanded_group:
                if idx not in current_group:
                    current_group.append(idx)
                    used_indices.add(idx)
        
        groups.append(current_group)
        
        if not silent:
            print(f"    Range group: {len(current_group)} sentences (similarity: {similarity:.3f})")
    
    return groups

def expand_group_in_range(sim_matrix, current_group, available_candidates, upper_threshold, lower_threshold, silent=True):
    """
    Expand a group by adding sentences that have similarity within the threshold range
    to at least one member of the current group.
    """
    expanded = []
    
    for candidate_idx in available_candidates:
        # Check if candidate has similarity within range to any group member
        max_similarity_to_group = -1
        for group_member_idx in current_group:
            similarity = sim_matrix[candidate_idx, group_member_idx]
            max_similarity_to_group = max(max_similarity_to_group, similarity)
        
        # Add to group if similarity is within the threshold range
        if lower_threshold <= max_similarity_to_group < upper_threshold:
            expanded.append(candidate_idx)
            if not silent:
                print(f"      Added sentence {candidate_idx} (max similarity: {max_similarity_to_group:.3f})")
    
    return expanded

def process_semantic_grouping_optimized(sim_matrix, sentences, initial_threshold, decay_factor, min_threshold, silent=True):
    """
    Wrapper function that delegates to the new threshold range algorithm.
    Maintains backward compatibility while using the new approach.
    """
    return process_semantic_grouping_with_threshold_ranges(
        sim_matrix, sentences, initial_threshold, decay_factor, min_threshold, silent
    )

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

def process_oie_extraction(chunk_text: str, max_triples: Optional[int] = None, silent: bool = True) -> Optional[str]:
    """Extract OIE relations for a chunk and format them"""
    if not chunk_text or not chunk_text.strip():
        return None
    
    try:
        if not silent:
            print(f"    Extracting OIE for chunk...")
        
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
    min_sentences_per_chunk: int = 1,  # Số câu tối thiểu mỗi chunk     
    include_oie: bool = False,
    save_raw_oie: bool = False,
    output_dir: str = "./output",
    initial_percentile: str = "85",  
    min_percentile: str = "25",
    embedding_batch_size: int = 64,  # Increased for RTX 5080
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
        print(f"   Min sentences per chunk: {min_sentences_per_chunk}")
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

    # For single or few sentences, keep as one chunk
    if len(sentences) <= 3:
        if not silent:
            print(f"   Small passage with {len(sentences)} sentences, keeping as single chunk")
        
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
        device=_normalize_device(kwargs.get('device', 'cuda')), 
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
        actual_min_threshold, silent
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
        
        num_sentences = len(valid_indices)
        
        # Check sentence count requirement only
        meets_sentence_requirement = num_sentences >= min_sentences_per_chunk
        
        if meets_sentence_requirement:
            final_chunks.append((chunk_id, final_chunk_text, oie_string))
            if not silent:
                print(f"   ✓ Chunk {i}: {num_sentences} sentences")
        else:
            if not silent:
                print(f"   ✗ Chunk {i}: {num_sentences} sentences < {min_sentences_per_chunk} (discarded)")

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

    # Save raw OIE if needed
    if save_raw_oie and all_raw_oie_data:
        try:
            raw_oie_filepath = helper_save_raw_oie_data(all_raw_oie_data, doc_id, output_dir, "semantic_grouping")
            if raw_oie_filepath and not silent:
                print(f"Raw OIE data saved to: {raw_oie_filepath}")
                total_relations = sum(entry['relation_count'] for entry in all_raw_oie_data)
                print(f"   Total chunks with OIE: {len(all_raw_oie_data)}")
                print(f"   Total relations extracted: {total_relations}")
        except Exception as e:
            if not silent:
                print(f"Warning: Failed to save raw OIE data: {e}")

    if not silent:
        print(f"  Created {len(final_chunks)} final chunks.")

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

# Compatibility wrapper to match interface expected by data_create_controller
def semantic_chunk_passage_from_grouping_logic(
    doc_id: str,
    passage_text: str,
    embedding_model: str = "thenlper/gte-base",
    initial_threshold: Union[str, float] = "auto",
    decay_factor: float = 0.85,
    min_threshold: Union[str, float] = "auto",
    window_size: int = 3,  # Unused in optimized version but kept for compatibility
    embedding_batch_size: int = 64,  # Increased for RTX 5080
    include_oie: bool = False,
    save_raw_oie: bool = False,
    output_dir: str = "./output",
    initial_percentile: str = "85",
    min_percentile: str = "25",
    target_tokens: int = 120,  # Ignored - no token constraints
    tolerance: float = 0.25,   # Ignored - no token constraints
    min_sentences_per_chunk: int = 1,  # Số câu tối thiểu mỗi chunk
    silent: bool = False,
    device: Optional[str] = "cuda",
    **kwargs
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Compatibility wrapper for the optimized semantic grouping.
    This function maintains the same interface as the original semantic_chunk_passage_from_grouping_logic
    but uses the optimized implementation without token constraints.
    """
    
    return semantic_grouping_main(
        passage_text=passage_text,
        doc_id=doc_id,
        embedding_model=embedding_model,
        initial_threshold=initial_threshold,
        decay_factor=decay_factor,
        min_threshold=min_threshold,
        min_sentences_per_chunk=min_sentences_per_chunk,
        include_oie=include_oie,
        save_raw_oie=save_raw_oie,
        output_dir=output_dir,
        initial_percentile=initial_percentile,
        min_percentile=min_percentile,
        embedding_batch_size=embedding_batch_size,
        silent=silent,
        device=device,
        **kwargs
    )
