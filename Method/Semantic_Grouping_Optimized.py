from dotenv import load_dotenv
from Tool.Sentence_Segmenter import extract_sentences_spacy
from .semantic_common import (
    normalize_device,
    embed_sentences_batched,
    create_similarity_matrix,
    analyze_similarity_distribution,
    log_msg,
)
import json
from typing import List, Tuple, Dict, Union, Optional
import gc
import numpy as np
import hashlib
import torch

load_dotenv()

## Removed local embedding + similarity helpers in favor of semantic_common

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
            log_msg(silent, f"Fast path small doc: {num_sentences} sentences", 'info', 'group')
            log_msg(silent, f"Single group formed ✓", 'info', 'group')
        return [list(range(num_sentences))]
    
    # Calculate sentence character lengths for size balancing
    sentence_lengths = [len(sentence) for sentence in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    
    # Estimate optimal sentences per chunk based on target size
    target_sentences_per_chunk = max(2, int(target_chunk_size / avg_sentence_length))
    # Remove minimum constraint, only set maximum to prevent oversized chunks
    max_chunk_size = int(target_chunk_size * (1 + size_tolerance))
    
    log_msg(silent, f"Chunk size ctrl: target={target_chunk_size} max={max_chunk_size} avg_sent={avg_sentence_length:.1f} target_sent_chunk={target_sentences_per_chunk}", 'debug', 'group')
    
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
    
    log_msg(silent, f"Start grouping: sentences={num_sentences} ranges={len(threshold_ranges)} {initial_threshold:.3f}->{min_threshold:.3f}", 'info', 'group')
    
    # Process each threshold range with size balancing
    for range_idx, (upper_threshold, lower_threshold) in enumerate(threshold_ranges):
        if len(ungrouped_indices) == 0:
            break
        log_msg(silent, f"Process range {range_idx+1}: [{upper_threshold:.3f},{lower_threshold:.3f}) remaining={len(ungrouped_indices)}", 'debug', 'group')
        # Direct iteration over remaining indices to form groups (simplified)
        range_groups = process_range_grouping_with_size_control(
            sim_matrix,
            sentences,
            sentence_lengths,
            list(ungrouped_indices),
            upper_threshold,
            lower_threshold,
            target_sentences_per_chunk,
            0,
            max_chunk_size,
            silent
        )
        for group in range_groups:
            if not group:
                continue
            groups.append(sorted(group))
            for idx in group:
                ungrouped_indices.discard(idx)
            group_chars = sum(sentence_lengths[idx] for idx in group)
            log_msg(silent, f"Group {len(groups)}: size={len(group)} chars={group_chars}", 'debug', 'group')
    
    # Handle remaining ungrouped sentences - PRESERVE ALL
    if ungrouped_indices:
        log_msg(silent, f"Preserve remaining sentences: {len(ungrouped_indices)}", 'debug', 'group')
        
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
        log_msg(silent, f"Grouping complete raw groups={len(groups)} (pre-split)", 'info', 'group')
        if groups:
            groups = split_oversized_chunks(groups, sentences, sentence_lengths, sim_matrix, max_chunk_size, silent)
            log_msg(silent, f"Post-split groups={len(groups)}", 'info', 'group')
        group_sizes = [sum(sentence_lengths[idx] for idx in g) for g in groups]
        if group_sizes:
            log_msg(silent, f"Size stats chars min={min(group_sizes)} max={max(group_sizes)} avg={sum(group_sizes)/len(group_sizes):.1f}", 'info', 'group')
    else:
        # Silent mode - still split oversized chunks
        if groups:
            groups = split_oversized_chunks(groups, sentences, sentence_lengths, sim_matrix, max_chunk_size, True)
        
    return groups

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
                log_msg(silent, f"Range group accepted size={len(group)} chars={group_chars}", 'debug', 'group')
    
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
                log_msg(silent, f"Semantic sub-chunk size={len(sub_group)} chars={sub_chars}", 'debug', 'group')
            
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
                log_msg(silent, f"Chunk {group_idx} ok size={len(group)} chars={group_chars}", 'debug', 'group')
        else:
            # Chunk quá lớn, cần chia nhỏ
            oversized_chunks.append((group_idx, group_chars))
            
            if not silent:
                log_msg(silent, f"Chunk {group_idx} oversize chars={group_chars} splitting", 'debug', 'group')
            
            # Ưu tiên chia theo ngữ nghĩa nếu có sim_matrix, fallback về chia tuần tự
            if sim_matrix is not None:
                sub_groups = smart_split_large_group(group, sentences, sentence_lengths, sim_matrix, max_chunk_size, silent)
            else:
                sub_groups = split_large_group(group, sentence_lengths, max_chunk_size, silent)
            final_groups.extend(sub_groups)
            split_count += 1
            
            if not silent:
                total_sub_chars = sum(sum(sentence_lengths[idx] for idx in sub_group) for sub_group in sub_groups)
                log_msg(silent, f"Split into {len(sub_groups)} sub-chunks total_chars={total_sub_chars}", 'debug', 'group')
    
    # Debug logging even in silent mode if splits occurred
    if split_count > 0 and not silent:
        log_msg(silent, f"Split {split_count} oversized chunks", 'info', 'group')
    
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
                    log_msg(silent, f"Seq sub-chunk size={len(current_sub_group)} chars={current_chars}", 'debug', 'group')
                
                # Bắt đầu sub-group mới
                current_sub_group = [sentence_idx]
                current_chars = sentence_chars
            else:
                # Trường hợp đặc biệt: câu đơn lẻ vượt quá giới hạn
                # Vẫn phải giữ lại để bảo toàn nội dung
                current_sub_group = [sentence_idx]
                current_chars = sentence_chars
                if not silent:
                    log_msg(silent, f"Single sentence exceeds limit chars={sentence_chars}", 'debug', 'group')
    
    # Thêm sub-group cuối cùng nếu có
    if current_sub_group:
        sub_groups.append(current_sub_group)
        if not silent:
            log_msg(silent, f"Seq sub-chunk size={len(current_sub_group)} chars={current_chars}", 'debug', 'group')
    
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
        log_msg(silent, f"Preserving remaining sentences count={len(remaining)}", 'debug', 'group')
    
    # Group remaining sentences, avoiding only oversized chunks
    while remaining:
        if len(remaining) <= target_sentences_per_chunk:
            # Last group - take all remaining to preserve content
            group_chars = sum(sentence_lengths[idx] for idx in remaining)
            if not silent:
                log_msg(silent, f"Final group size={len(remaining)} chars={group_chars}", 'debug', 'group')
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
                log_msg(silent, f"Preserved group size={len(current_group)} chars={current_chars}", 'debug', 'group')
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

## Removed local OIE formatting & extraction (using semantic_common functions)

def semantic_grouping_main(
    passage_text: str,
    doc_id: str,
    embedding_model: str,  
    initial_threshold: Union[str, float] = "auto",
    decay_factor: float = 0.85,
    min_threshold: Union[str, float] = "auto",
    min_sentences_per_chunk: int = 1,  # Số câu tối thiểu mỗi chunk     
    include_oie: bool = False,  # retained for backward compatibility – ignored
    save_raw_oie: bool = False, # retained – ignored
    output_dir: str = "./output",
    initial_percentile: str = "85",  
    min_percentile: str = "25",
    embedding_batch_size: int = 64,  
    # New balanced chunking parameters
    enable_balanced_chunking: bool = False,  # Enable balanced chunking algorithm
    target_chunk_size: int = 150,           # Target character count per chunk
    size_tolerance: float = 0.5,            # Tolerance for chunk size deviation (0.5 = ±50%)
    # Sentence-based balancing (context-preserve) parameters
    target_sentences_per_chunk: Optional[int] = None,  # Mục tiêu số câu / chunk (ưu tiên hơn target_chunk_size nếu thiết lập)
    enforce_sentence_balancing: bool = True,  # Bật hợp nhất / chia dựa số câu với bảo toàn ngữ cảnh
    silent: bool = False,
    **kwargs
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Enhanced semantic grouping with improved algorithm and OIE support.
    
    Returns:
        List of tuples: (chunk_id, chunk_text_with_oie, oie_string_only)
    """

    if include_oie and not silent:
        log_msg(silent, "OIE generation disabled in grouping (handled externally)", 'debug', 'group')
    if not silent:
        log_msg(silent, f"Semantic grouping doc={doc_id} model={embedding_model} min_sent={min_sentences_per_chunk}", 'info', 'group')

    # Extract sentences
    sentences = extract_sentences_spacy(passage_text)
    if not sentences:
        if not silent:
            log_msg(silent, f"No sentences found doc={doc_id}", 'warning', 'group')
        
    return [(f"{doc_id}_single_empty", passage_text, None)]

    # For single or few sentences, keep as one chunk
    if len(sentences) <= 3:
        if not silent:
            log_msg(silent, f"Small passage sentences={len(sentences)} keep single chunk", 'info', 'group')
        
    return [(f"{doc_id}_single_complete", passage_text, None)]

    # Create similarity matrix
    if not silent:
        log_msg(silent, f"Create similarity matrix sentences={len(sentences)}", 'info', 'group')
    
    sim_matrix = create_similarity_matrix(
        sentences=sentences,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=normalize_device(kwargs.get('device', 'cuda')),
        silent=silent,
    )
    
    if sim_matrix is None:
        if not silent:
            log_msg(silent, f"Failed to create similarity matrix", 'error', 'group')
        
    return [(f"{doc_id}_matrix_fail", passage_text, None)]

    # Resolve auto thresholds
    actual_initial_threshold = initial_threshold
    actual_min_threshold = min_threshold
    
    if initial_threshold == 'auto' or min_threshold == 'auto':
        if not silent:
            log_msg(silent, "Auto-threshold analyzing distribution", 'info', 'group')
        
        dist_stats = analyze_similarity_distribution(sim_matrix)
        if dist_stats:
            if initial_threshold == 'auto':
                percentile_key = f'p{initial_percentile}'
                actual_initial_threshold = dist_stats.get(percentile_key, 0.75)
                if not silent:
                    log_msg(silent, f"Auto initial_threshold p{initial_percentile}={actual_initial_threshold:.4f}", 'info', 'group')
            
            if min_threshold == 'auto':
                percentile_key = f'p{min_percentile}'
                actual_min_threshold = dist_stats.get(percentile_key, 0.5)
                if not silent:
                    log_msg(silent, f"Auto min_threshold p{min_percentile}={actual_min_threshold:.4f}", 'info', 'group')
            
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
                log_msg(silent, "Distribution analysis failed, using defaults", 'warning', 'group')
            actual_initial_threshold = 0.75 if initial_threshold == 'auto' else float(initial_threshold)
            actual_min_threshold = 0.5 if min_threshold == 'auto' else float(min_threshold)
    else:
        actual_initial_threshold = float(initial_threshold)
        actual_min_threshold = float(min_threshold)

    # Perform semantic grouping
    if not silent:
        log_msg(silent, f"Grouping thresholds {actual_initial_threshold:.4f}->{actual_min_threshold:.4f}", 'info', 'group')
        if enable_balanced_chunking:
            log_msg(silent, f"Balanced chunking target={target_chunk_size} tol={int(size_tolerance*100)}%", 'info', 'group')
    
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

    # --- Sentence-level balancing & consolidation (context-preserve) ---
    if enforce_sentence_balancing and group_indices:
        group_indices = consolidate_and_balance_groups(
            sim_matrix=sim_matrix,
            groups=group_indices,
            min_sentences=min_sentences_per_chunk,
            target_sentences=target_sentences_per_chunk,
            sentences=sentences,
            silent=silent
        )
    
    del sim_matrix
    gc.collect()

    if not group_indices:
        if not silent:
            log_msg(silent, "No groups formed fallback original", 'warning', 'group')
        return [(f"{doc_id}_no_groups", passage_text, None)]

    # Create final chunks with OIE
    if not silent:
        log_msg(silent, f"Create final chunks count={len(group_indices)} OIE={include_oie}", 'info', 'group')
    
    final_chunks: List[Tuple[str, str, Optional[str]]] = []
    
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

        oie_string = None  # placeholder after OIE removal
        num_sentences = len(valid_indices)
        final_chunks.append((chunk_id, chunk_text, oie_string))
        if not silent:
            log_msg(silent, f"Chunk {i} preserved sentences={num_sentences}", 'debug', 'group')

    # Handle case where no valid chunks were created
    if not final_chunks:
        if not silent:
            log_msg(silent, "No valid chunks after filtering fallback", 'warning', 'group')
        final_chunks = [(f"{doc_id}_filter_fail", passage_text, None)]

    # Save raw OIE if needed
    # Raw OIE saving removed (handled externally)

    if not silent:
        log_msg(silent, f"Created final chunks={len(final_chunks)}", 'info', 'group')

    return final_chunks

## Removed local helper_save_raw_oie_data in favor of save_raw_oie_data from semantic_common

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
    target_sentences_per_chunk: Optional[int] = None,
    enforce_sentence_balancing: bool = True,
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
        target_sentences_per_chunk=target_sentences_per_chunk,
        enforce_sentence_balancing=enforce_sentence_balancing,
        **kwargs
    )

# ================= Added: Consolidate & Balance Groups ==================
def consolidate_and_balance_groups(
    sim_matrix: np.ndarray,
    groups: List[List[int]],
    min_sentences: int,
    target_sentences: Optional[int],
    sentences: List[str],
    silent: bool = True,
) -> List[List[int]]:
    """Consolidate small groups (< min_sentences) by merging with the most semantically similar
    neighbor; optionally split very large groups using semantic cut points toward target_sentences.

    Steps:
      1. Sort groups by first sentence index to preserve document order.
      2. Merge any group whose size < min_sentences with whichever adjacent group (prev/next) yields
         higher average cross similarity.
      3. If target_sentences is set: split groups whose size > 2 * target_sentences into near-equal
         semantic segments using internal similarity.
      4. Re-run a final pass to ensure no residual group < min_sentences (merge forward).
    """
    if not groups:
        return groups
    if min_sentences <= 1 and target_sentences is None:
        return groups

    # Order groups by natural sentence order
    ordered = sorted(groups, key=lambda g: min(g) if g else 10**9)

    def group_similarity(g1: List[int], g2: List[int]) -> float:
        if not g1 or not g2:
            return -1.0
        sims = sim_matrix[np.ix_(g1, g2)]
        if sims.size == 0:
            return -1.0
        return float(np.mean(sims))

    # Merge small groups
    changed = True
    while changed:
        changed = False
        new_ordered: List[List[int]] = []
        i = 0
        while i < len(ordered):
            grp = ordered[i]
            if len(grp) >= min_sentences or len(ordered) == 1:
                new_ordered.append(grp)
                i += 1
                continue
            # Need merge: pick neighbor with higher avg sim
            prev_grp = ordered[i-1] if i > 0 else None
            next_grp = ordered[i+1] if i+1 < len(ordered) else None
            sim_prev = group_similarity(prev_grp, grp) if prev_grp else -1
            sim_next = group_similarity(grp, next_grp) if next_grp else -1
            if sim_prev < 0 and sim_next < 0:
                # Fallback: merge forward if possible else backward
                if next_grp is not None:
                    merged = grp + next_grp
                    new_ordered.append(merged)
                    i += 2
                elif prev_grp is not None:
                    merged = prev_grp + grp
                    # Replace previously added prev_grp
                    if new_ordered:
                        new_ordered[-1] = merged
                    i += 1
                else:
                    new_ordered.append(grp)
                    i += 1
            elif sim_prev >= sim_next:
                # Merge into prev
                if new_ordered:
                    new_ordered[-1] = new_ordered[-1] + grp
                else:
                    # Shouldn't happen because prev exists, but guard
                    new_ordered.append(grp)
                i += 1
            else:
                # Merge with next
                if next_grp is not None:
                    merged = grp + next_grp
                    new_ordered.append(merged)
                    i += 2
                else:
                    # No next, merge backward
                    if new_ordered:
                        new_ordered[-1] = new_ordered[-1] + grp
                    else:
                        new_ordered.append(grp)
                    i += 1
            changed = True
        ordered = [sorted(g) for g in new_ordered]

    # Optional splitting of very large groups
    if target_sentences and target_sentences > 0:
        adjusted: List[List[int]] = []
        upper = 2 * target_sentences
        for g in ordered:
            if len(g) > upper:
                # Determine split count
                k = max(1, round(len(g) / target_sentences))
                # Avoid making subgroups < min_sentences
                while k > 1 and len(g) / k < min_sentences:
                    k -= 1
                if k == 1:
                    adjusted.append(g)
                    continue
                # Simple proportional split boundaries
                boundaries = [0]
                for i in range(1, k):
                    boundaries.append(round(i * len(g) / k))
                boundaries.append(len(g))
                for b in range(len(boundaries)-1):
                    sub = g[boundaries[b]:boundaries[b+1]]
                    if sub:
                        adjusted.append(sub)
                if not silent:
                    print(f"   Split large group ({len(g)} sentences) → {len(boundaries)-1} sub-groups")
            else:
                adjusted.append(g)
        ordered = adjusted

    # Final pass: if last group still < min_sentences and >1 group, merge backward
    if ordered and len(ordered) > 1 and len(ordered[-1]) < min_sentences:
        ordered[-2].extend(ordered[-1])
        ordered.pop()

    if not silent:
        stats = [len(g) for g in ordered]
        if stats:
            print(f"   Consolidation complete: groups={len(ordered)}, size range={min(stats)}-{max(stats)}, avg={sum(stats)/len(stats):.2f}")
    return ordered
