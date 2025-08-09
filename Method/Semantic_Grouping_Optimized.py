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
import torch

load_dotenv()

def _normalize_device(device) -> str:
    """Normalize device to 'cuda' or 'cpu', with automatic fallback if CUDA is unavailable."""
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    device_str = str(device).lower() if device is not None else None
    if device_str in (None, "auto"):
        return "cuda" if cuda_available else "cpu"
    if device_str.startswith("cuda"):
        return "cuda" if cuda_available else "cpu"
    return "cpu"

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

        # Adaptive sub-batching to avoid OOM
        internal_batch_size = len(batch) if device == "cuda" else min(len(batch), 16)
        cur_bs = max(1, internal_batch_size)
        start_idx = 0
        batch_embeds_list = []
        while start_idx < len(batch):
            end_idx = min(start_idx + cur_bs, len(batch))
            sub_batch = batch[start_idx:end_idx]
            try:
                sub_embeddings = embed_text_list(
                    sub_batch,
                    model_name=model_name,
                    batch_size=cur_bs,
                    device_preference=device
                )
                if sub_embeddings is not None and len(sub_embeddings) > 0:
                    batch_embeds_list.append(sub_embeddings)
                    start_idx = end_idx
                else:
                    if not silent:
                        print(f"Warning: Sub-batch returned no embeddings. Indices {start_idx}:{end_idx}")
                    start_idx = end_idx
            except Exception as e:
                message = str(e).lower()
                is_oom = ("out of memory" in message) or ("cuda" in message and "memory" in message) or (hasattr(torch, "cuda") and isinstance(e, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)))
                if is_oom and device == "cuda":
                    if cur_bs <= 1:
                        if not silent:
                            print(f"OOM at batch {batch_num} even with batch_size=1, skipping these items.")
                        start_idx = end_idx
                        continue
                    cur_bs = max(1, cur_bs // 2)
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if not silent:
                        print(f"    OOM detected. Reducing sub-batch size to {cur_bs} and retrying...")
                    continue
                else:
                    if not silent:
                        print(f"Error during embedding at batch {batch_num}: {e}")
                        print(f"Device: {device}, Attempted sub-batch size: {cur_bs}")
                    # Skip this sub-batch to avoid blocking entire run
                    start_idx = end_idx
                    continue
        if batch_embeds_list:
            try:
                all_embeddings.append(np.vstack(batch_embeds_list))
            except Exception:
                concatenated = []
                for be in batch_embeds_list:
                    concatenated.extend(list(be))
                all_embeddings.append(np.array(concatenated))

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
    """
    GPU-Optimized semantic similarity matrix creation.
    Uses PyTorch GPU tensors for faster matrix operations.
    """
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
        # **GPU-ACCELERATED SIMILARITY COMPUTATION**
        device_obj = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        
        if not silent:
            print(f"   Computing similarity matrix on {device_obj}...")
        
        matrix_start = time.time()
        
        if device_obj.type == "cuda":
            # GPU-accelerated computation using PyTorch
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device_obj)
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
            
            # Compute similarity matrix using GPU matrix multiplication (MUCH faster)
            sim_matrix_tensor = torch.mm(embeddings_normalized, embeddings_normalized.t())
            
            # Convert back to numpy for compatibility
            sim_matrix = sim_matrix_tensor.cpu().numpy()
            
            # Clean up GPU memory
            del embeddings_tensor, embeddings_normalized, sim_matrix_tensor
            torch.cuda.empty_cache()
            
        else:
            # CPU fallback using sklearn
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            sim_matrix = cosine_similarity(embeddings)
        
        matrix_time = time.time() - matrix_start
        total_time = time.time() - start_time
        
        if not silent:
            print(f"   Similarity matrix computed in {matrix_time:.2f}s on {device_obj}")
            print(f"   Total time: {total_time:.2f}s | Matrix shape: {sim_matrix.shape}")
        
        del embeddings
        gc.collect()
        return sim_matrix
        
    except Exception as e:
        if not silent:
            print(f"GPU similarity computation failed, falling back to CPU: {e}")
        
        # Fallback to original CPU method
        try:
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            sim_matrix = cosine_similarity(embeddings)
            del embeddings
            gc.collect()
            return sim_matrix
        except Exception as fallback_e:
            if not silent:
                print(f"Error calculating similarity matrix: {fallback_e}")
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

def process_semantic_grouping_with_balanced_chunks(sim_matrix, sentences, initial_threshold, decay_factor, min_threshold, target_chunk_size=150, size_tolerance=0.5, silent=True):
    """
    Enhanced semantic grouping with chunk size balancing.
    MODIFIED: Now preserves ALL content, only controls maximum chunk sizes.
    
    Args:
        target_chunk_size: Target number of characters per chunk
        size_tolerance: Tolerance for chunk size deviation (0.5 = ±50%)
    
    Process chunks in descending threshold ranges while preventing oversized chunks:
    - Range 1: [initial_threshold, initial_threshold*decay_factor] 
    - Range 2: [initial_threshold*decay_factor, initial_threshold*decay_factor^2]
    - ... but with upper size constraints to avoid very large chunks
    """
    num_sentences = sim_matrix.shape[0]
    
    # Fast path for very small documents (≤6 sentences)
    if num_sentences <= 6:
        if not silent:
            print(f"Fast path for small document ({num_sentences} sentences)")
            print(f"  Single group: {num_sentences} sentences ✓")
        return [list(range(num_sentences))]
    
    # Calculate sentence character lengths for size balancing
    sentence_lengths = [len(sentence) for sentence in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    
    # Estimate optimal sentences per chunk based on target size
    target_sentences_per_chunk = max(2, int(target_chunk_size / avg_sentence_length))
    # Remove minimum constraint, only set maximum to prevent oversized chunks
    max_chunk_size = int(target_chunk_size * (1 + size_tolerance))
    
    if not silent:
        print(f" Chunk Size Control (No Minimum Constraints):")
        print(f"   Target chunk size: {target_chunk_size} chars")
        print(f"   Maximum chunk size: {max_chunk_size} chars (no minimum)")
        print(f"   Avg sentence length: {avg_sentence_length:.1f} chars")
        print(f"   Target sentences per chunk: {target_sentences_per_chunk}")
    
    ungrouped_indices = set(range(num_sentences))
    groups = []
    
    # Calculate threshold ranges
    threshold_ranges = []
    current_upper = initial_threshold
    range_index = 0
    
    while current_upper >= min_threshold and range_index < 15:  # Reduce max ranges
        current_lower = max(min_threshold, current_upper * decay_factor)
        threshold_ranges.append((current_upper, current_lower))
        current_upper = current_lower
        range_index += 1
        
        if current_lower <= min_threshold:
            break
    
    if not silent:
        print(f"Starting content-preserving threshold range grouping with {num_sentences} sentences")
        print(f"Threshold ranges: {len(threshold_ranges)} ranges from {initial_threshold:.3f} to {min_threshold:.3f}")
    
    # Process each threshold range with size balancing
    for range_idx, (upper_threshold, lower_threshold) in enumerate(threshold_ranges):
        if len(ungrouped_indices) == 0:
            break
            
        if not silent:
            print(f"\n--- Processing Range {range_idx + 1}: [{upper_threshold:.3f}, {lower_threshold:.3f}) ---")
        
        range_groups = process_range_grouping_with_size_control(
            sim_matrix, 
            sentences,
            sentence_lengths,
            list(ungrouped_indices),
            upper_threshold, 
            lower_threshold,
            target_sentences_per_chunk,
            0,  # No minimum size constraint
            max_chunk_size,
            silent
        )
        
        # Add valid groups and remove grouped sentences
        for group in range_groups:
            if group:
                groups.append(sorted(group))
                for idx in group:
                    ungrouped_indices.discard(idx)
                    
                # Calculate actual chunk size
                group_chars = sum(sentence_lengths[idx] for idx in group)
                if not silent:
                    print(f"  Group {len(groups)}: {len(group)} sentences, {group_chars} chars")
    
    # Handle remaining ungrouped sentences - PRESERVE ALL
    if ungrouped_indices:
        if not silent:
            print(f"\n--- Preserving {len(ungrouped_indices)} remaining sentences ---")
        
        remaining_groups = balance_remaining_sentences(
            list(ungrouped_indices), 
            sentences, 
            sentence_lengths,
            target_sentences_per_chunk,
            0,  # No minimum size constraint
            max_chunk_size,
            silent
        )
        groups.extend(remaining_groups)
    
    if not silent:
        print(f"\nContent-preserving grouping completed: {len(groups)} groups formed")
        
        # Split oversized chunks into smaller ones
        if groups:
            print(f"\n--- Splitting oversized chunks (max: {max_chunk_size} chars) ---")
            groups = split_oversized_chunks(groups, sentences, sentence_lengths, sim_matrix, max_chunk_size, silent)
            print(f"Final result: {len(groups)} chunks after splitting")
        
        # Print size statistics
        group_sizes = []
        for group in groups:
            group_chars = sum(sentence_lengths[idx] for idx in group)
            group_sizes.append(group_chars)
        
        if group_sizes:
            print(f"Chunk size stats: min={min(group_sizes)}, max={max(group_sizes)}, avg={sum(group_sizes)/len(group_sizes):.1f}")
    else:
        # Silent mode - still split oversized chunks
        if groups:
            groups = split_oversized_chunks(groups, sentences, sentence_lengths, sim_matrix, max_chunk_size, True)
        
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

def process_range_grouping_with_size_control(sim_matrix, sentences, sentence_lengths, available_indices, 
                                          upper_threshold, lower_threshold, target_sentences_per_chunk,
                                          min_chunk_size, max_chunk_size, silent=True):
    """
    Process grouping within a threshold range with size control.
    Now accepts ALL groups to avoid losing content.
    """
    if not available_indices:
        return []
    
    groups = []
    visited = set()
    
    for start_idx in available_indices:
        if start_idx in visited:
            continue
        
        # Find connected component using Union-Find within threshold range
        group = expand_group_with_size_control(
            sim_matrix, sentence_lengths, available_indices, start_idx,
            lower_threshold, upper_threshold, target_sentences_per_chunk,
            max_chunk_size, visited
        )
        
        if group:
            group_chars = sum(sentence_lengths[idx] for idx in group)
            
            # Accept ALL groups to preserve content - no filtering by minimum size
            groups.append(group)
            visited.update(group)
            
            if not silent:
                print(f"    Group: {len(group)} sentences, {group_chars} chars ✓")
    
    return groups

def expand_group_with_size_control(sim_matrix, sentence_lengths, available_indices, start_idx, 
                                 lower_threshold, upper_threshold, target_sentences, max_chars, visited):
    """
    Expand group from start_idx within threshold range with size constraints.
    Now focuses only on preventing groups that are TOO LARGE, not filtering small ones.
    """
    if start_idx in visited:
        return []
    
    group = [start_idx]
    group_chars = sentence_lengths[start_idx]
    candidates = set(available_indices) - visited - {start_idx}
    
    # Greedy expansion with upper size limit only
    while candidates and len(group) < target_sentences * 3:  # Allow more flexibility for larger groups
        best_candidate = None
        best_similarity = 0
        
        # Find best candidate that doesn't exceed MAXIMUM size limit
        for candidate in candidates:
            candidate_chars = sentence_lengths[candidate]
            
            # Only check if adding this candidate would exceed MAXIMUM size limit
            # Remove minimum constraints to preserve content
            if group_chars + candidate_chars > max_chars:
                continue  # Only skip if group becomes TOO LARGE
                
            # Find maximum similarity with any sentence in current group
            max_sim = 0
            for group_member in group:
                sim = sim_matrix[group_member, candidate]
                if lower_threshold <= sim < upper_threshold:
                    max_sim = max(max_sim, sim)
            
            # Select candidate with highest similarity
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_candidate = candidate
        
        # Add best candidate if found
        if best_candidate and best_similarity > 0:
            group.append(best_candidate)
            group_chars += sentence_lengths[best_candidate]
            candidates.remove(best_candidate)
            
            # Early stop if we reach reasonable upper size limit
            if group_chars >= max_chars * 0.9:  # Stop at 90% of max to avoid getting too close
                break
        else:
            break  # No more valid candidates
    
    return group

def smart_split_large_group(group, sentences, sentence_lengths, sim_matrix, max_chunk_size, silent=True):
    """
    Chia một group lớn thành các group nhỏ hơn dựa trên semantic similarity.
    
    Chiến lược: Tìm điểm cắt có similarity thấp nhất để chia group một cách tự nhiên
    """
    if len(group) <= 1:
        return [group]
    
    # Sắp xếp group theo thứ tự ban đầu
    sorted_group = sorted(group)
    
    # Nếu group nhỏ, sử dụng chia tuần tự đơn giản
    if len(sorted_group) <= 3:
        return split_large_group(group, sentence_lengths, max_chunk_size, silent)
    
    # Tìm các điểm cắt tốt dựa trên similarity
    cut_points = find_semantic_cut_points(sorted_group, sim_matrix, sentence_lengths, max_chunk_size)
    
    if not cut_points:
        # Fallback về chia tuần tự
        return split_large_group(group, sentence_lengths, max_chunk_size, silent)
    
    # Chia group tại các điểm cắt
    sub_groups = []
    start_idx = 0
    
    for cut_point in cut_points + [len(sorted_group)]:  # Thêm điểm cuối
        if cut_point > start_idx:
            sub_group = sorted_group[start_idx:cut_point]
            sub_groups.append(sub_group)
            
            if not silent:
                sub_chars = sum(sentence_lengths[idx] for idx in sub_group)
                print(f"        Semantic sub-chunk: {len(sub_group)} sentences, {sub_chars} chars")
            
            start_idx = cut_point
    
    return sub_groups

def find_semantic_cut_points(sorted_group, sim_matrix, sentence_lengths, max_chunk_size):
    """
    Tìm các điểm cắt tốt dựa trên semantic similarity giữa các câu liên tiếp.
    """
    if len(sorted_group) <= 2:
        return []
    
    # Tính similarity giữa các câu liên tiếp
    similarities = []
    for i in range(len(sorted_group) - 1):
        idx1, idx2 = sorted_group[i], sorted_group[i + 1]
        sim = sim_matrix[idx1, idx2] if sim_matrix is not None else 0.5
        similarities.append((i + 1, sim))  # (cut_position, similarity)
    
    # Sắp xếp theo similarity tăng dần (similarity thấp = điểm cắt tốt)
    similarities.sort(key=lambda x: x[1])
    
    # Chọn điểm cắt tốt nhất mà vẫn đảm bảo kích thước
    cut_points = []
    current_start = 0
    
    for cut_pos, sim in similarities:
        # Kiểm tra xem cắt tại đây có tạo chunk hợp lý không
        chunk1 = sorted_group[current_start:cut_pos]
        chunk1_chars = sum(sentence_lengths[idx] for idx in chunk1)
        
        # Chỉ cắt nếu chunk đầu không quá nhỏ và không vượt quá giới hạn
        if len(chunk1) >= 2 and chunk1_chars <= max_chunk_size:
            # Kiểm tra phần còn lại có cần chia tiếp không
            remaining = sorted_group[cut_pos:]
            remaining_chars = sum(sentence_lengths[idx] for idx in remaining)
            
            if remaining_chars > max_chunk_size:
                # Cần chia tiếp, chấp nhận điểm cắt này
                cut_points.append(cut_pos)
                current_start = cut_pos
            else:
                # Phần còn lại đã OK, có thể dừng
                break
    
    return cut_points

def split_oversized_chunks(groups, sentences, sentence_lengths, sim_matrix, max_chunk_size, silent=True):
    """
    Chia các chunk quá lớn thành các chunk nhỏ hơn.
    
    Args:
        groups: List các group indices
        sentences: List các câu
        sentence_lengths: List độ dài từng câu
        sim_matrix: Ma trận tương đồng để hỗ trợ chia theo ngữ nghĩa
        max_chunk_size: Kích thước tối đa cho phép
        silent: Có in log không
    
    Returns:
        List các group đã được chia nhỏ
    """
    if not groups:
        return groups
    
    final_groups = []
    split_count = 0
    oversized_chunks = []
    
    for group_idx, group in enumerate(groups):
        group_chars = sum(sentence_lengths[idx] for idx in group)
        
        if group_chars <= max_chunk_size:
            # Chunk đã OK, giữ nguyên
            final_groups.append(group)
            if not silent:
                print(f"    Chunk {group_idx}: {len(group)} sentences, {group_chars} chars ✓")
        else:
            # Chunk quá lớn, cần chia nhỏ
            oversized_chunks.append((group_idx, group_chars))
            
            if not silent:
                print(f"    Chunk {group_idx}: {len(group)} sentences, {group_chars} chars - SPLITTING...")
            
            # Ưu tiên chia theo ngữ nghĩa nếu có sim_matrix, fallback về chia tuần tự
            if sim_matrix is not None:
                sub_groups = smart_split_large_group(group, sentences, sentence_lengths, sim_matrix, max_chunk_size, silent)
            else:
                sub_groups = split_large_group(group, sentence_lengths, max_chunk_size, silent)
            final_groups.extend(sub_groups)
            split_count += 1
            
            if not silent:
                total_sub_chars = sum(sum(sentence_lengths[idx] for idx in sub_group) for sub_group in sub_groups)
                print(f"      → Split into {len(sub_groups)} sub-chunks, total: {total_sub_chars} chars")
    
    # Debug logging even in silent mode if splits occurred
    if split_count > 0 and not silent:
        print(f"  Split {split_count} oversized chunks into smaller ones")
    
    return final_groups

def split_large_group(group, sentence_lengths, max_chunk_size, silent=True):
    """
    Chia một group lớn thành các group nhỏ hơn.
    
    Chiến lược: Chia tuần tự, đảm bảo mỗi sub-group không vượt quá max_chunk_size
    """
    if not group:
        return []
    
    # Sắp xếp group theo thứ tự ban đầu để duy trì tính liên tục
    sorted_group = sorted(group)
    sub_groups = []
    current_sub_group = []
    current_chars = 0
    
    for sentence_idx in sorted_group:
        sentence_chars = sentence_lengths[sentence_idx]
        
        # Kiểm tra nếu thêm câu này có vượt quá giới hạn không
        if current_chars + sentence_chars <= max_chunk_size:
            # An toàn để thêm
            current_sub_group.append(sentence_idx)
            current_chars += sentence_chars
        else:
            # Vượt quá giới hạn
            if current_sub_group:
                # Lưu sub-group hiện tại
                sub_groups.append(current_sub_group.copy())
                if not silent:
                    print(f"        Sub-chunk: {len(current_sub_group)} sentences, {current_chars} chars")
                
                # Bắt đầu sub-group mới
                current_sub_group = [sentence_idx]
                current_chars = sentence_chars
            else:
                # Trường hợp đặc biệt: câu đơn lẻ vượt quá giới hạn
                # Vẫn phải giữ lại để bảo toàn nội dung
                current_sub_group = [sentence_idx]
                current_chars = sentence_chars
                if not silent:
                    print(f" Single sentence exceeds limit: {sentence_chars} chars")
    
    # Thêm sub-group cuối cùng nếu có
    if current_sub_group:
        sub_groups.append(current_sub_group)
        if not silent:
            print(f"        Sub-chunk: {len(current_sub_group)} sentences, {current_chars} chars")
    
    return sub_groups

def balance_remaining_sentences(remaining_indices, sentences, sentence_lengths, 
                              target_sentences_per_chunk, min_chunk_size, max_chunk_size, silent=True):
    """
    Balance remaining ungrouped sentences into appropriately sized chunks.
    MODIFIED: Now preserves ALL sentences, no minimum size filtering.
    """
    if not remaining_indices:
        return []
    
    groups = []
    remaining = remaining_indices.copy()
    
    if not silent:
        print(f"    Preserving {len(remaining)} remaining sentences (no content loss)")
    
    # Group remaining sentences, avoiding only oversized chunks
    while remaining:
        if len(remaining) <= target_sentences_per_chunk:
            # Last group - take all remaining to preserve content
            group_chars = sum(sentence_lengths[idx] for idx in remaining)
            if not silent:
                print(f"    Final group: {len(remaining)} sentences, {group_chars} chars ✓")
            groups.append(remaining.copy())
            break
        
        # Create group up to max size limit
        current_group = []
        current_chars = 0
        
        # Take sentences up to maximum size limit (not minimum)
        for idx in remaining:
            sentence_chars = sentence_lengths[idx]
            
            # Only check if adding would exceed MAXIMUM limit
            if current_chars + sentence_chars <= max_chunk_size:
                current_group.append(idx)
                current_chars += sentence_chars
            else:
                # If current group has at least one sentence, stop here
                if current_group:
                    break
                else:
                    # If single sentence exceeds max, take it anyway to preserve content
                    current_group.append(idx)
                    current_chars += sentence_chars
                    break
        
        if current_group:
            groups.append(current_group)
            remaining = [idx for idx in remaining if idx not in current_group]
            
            if not silent:
                print(f"    Preserved group: {len(current_group)} sentences, {current_chars} chars ✓")
        else:
            # Fallback - take first sentence to avoid infinite loop
            groups.append([remaining[0]])
            remaining = remaining[1:]
    
    return groups

def process_semantic_grouping_optimized(sim_matrix, sentences, initial_threshold, decay_factor, min_threshold, silent=True):
    """
    ORIGINAL semantic grouping algorithm without size constraints.
    
    This function maintains backward compatibility for true "no limit" semantic grouping.
    Uses range-based grouping with similarity thresholds only, no size balancing.
    """
    # Call the original range-based algorithm from Semantic_Grouping.py
    from Method.Semantic_Grouping import semantic_spreading_grouping_optimized
    
    return semantic_spreading_grouping_optimized(
        sim_matrix, sentences, initial_threshold, decay_factor, min_threshold, 
        min_chunk_len=0, max_chunk_len=float('inf'), silent=silent
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
    embedding_batch_size: int = 64,  
    # New balanced chunking parameters
    enable_balanced_chunking: bool = False,  # Enable balanced chunking algorithm
    target_chunk_size: int = 150,           # Target character count per chunk
    size_tolerance: float = 0.5,            # Tolerance for chunk size deviation (0.5 = ±50%)
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
        if enable_balanced_chunking:
            print(f"   Balanced chunking enabled - target: {target_chunk_size} chars, tolerance: ±{int(size_tolerance*100)}%")
    
    # Choose algorithm based on parameters
    if enable_balanced_chunking:
        group_indices = process_semantic_grouping_with_balanced_chunks(
            sim_matrix, sentences, actual_initial_threshold, decay_factor, 
            actual_min_threshold, target_chunk_size, size_tolerance, silent
        )
    else:
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
        
        # Check sentence count requirement only - ACCEPT ALL CHUNKS to preserve content
        meets_sentence_requirement = True  # Accept all chunks regardless of size
        
        if meets_sentence_requirement:
            final_chunks.append((chunk_id, final_chunk_text, oie_string))
            if not silent:
                print(f"   ✓ Chunk {i}: {num_sentences} sentences (preserved)")
        else:
            # This branch should never be reached now, but keep for safety
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
    # New balanced chunking parameters
    enable_balanced_chunking: bool = False,  # Enable balanced chunking algorithm
    target_chunk_size: int = 150,           # Target character count per chunk
    size_tolerance: float = 0.5,            # Tolerance for chunk size deviation (0.5 = ±50%)
    silent: bool = False,
    device: Optional[str] = "cuda",
    **kwargs
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Compatibility wrapper for the optimized semantic grouping.
    This function maintains the same interface as the original semantic_chunk_passage_from_grouping_logic
    but uses the optimized implementation.
    
    New Parameters:
        enable_balanced_chunking: If True, uses the balanced algorithm that controls chunk sizes
        target_chunk_size: Target number of characters per chunk (when balanced chunking enabled)
        size_tolerance: Tolerance for chunk size deviation (0.5 = ±50%)
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
        enable_balanced_chunking=enable_balanced_chunking,
        target_chunk_size=target_chunk_size,
        size_tolerance=size_tolerance,
        silent=silent,
        device=device,
        **kwargs
    )
