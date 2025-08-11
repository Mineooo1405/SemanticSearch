from dotenv import load_dotenv
from Tool.Sentence_Segmenter import extract_sentences_spacy
from .semantic_common import (
    normalize_device,
    create_similarity_matrix,
    analyze_similarity_distribution,
    log_msg,
)
from typing import List, Tuple, Union, Optional
import gc
import numpy as np
import hashlib

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
    # Sentence-based parameters ONLY (character size control removed)
    target_sentences_per_chunk: Optional[int] = None,
    enforce_sentence_balancing: bool = True,
    sentence_target_tolerance: float = 0.3,
    max_sentences_per_chunk: Optional[int] = None,  # NEW: hard upper cap (split if exceeded)
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

    # Choose algorithm based on sentence target only
    if (target_sentences_per_chunk is not None) and (target_sentences_per_chunk > 0):
        # Pure sentence-count controlled path (IGNORE character lengths)
        group_indices = sentence_count_grouping(
            sim_matrix=sim_matrix,
            sentences=sentences,
            target_sentences=target_sentences_per_chunk,
            min_sentences=min_sentences_per_chunk,
            tolerance=sentence_target_tolerance,
            upper_threshold=actual_initial_threshold,
            lower_threshold=actual_min_threshold,
            decay_factor=decay_factor,
            silent=silent
        )
    else:
        # Original pure semantic grouping (no size constraints)
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

    # Additional pass: merge excessively small groups toward target_sentences_per_chunk
    if target_sentences_per_chunk and target_sentences_per_chunk > 0 and group_indices:
        group_indices = merge_small_groups_toward_sentence_target(
            group_indices,
            target_sentences_per_chunk,
            sentence_target_tolerance,
            min_sentences_per_chunk,
            silent
        )

    # --- Enforce hard max sentences per chunk (simple sequential split) ---
    if max_sentences_per_chunk and max_sentences_per_chunk > 0 and group_indices:
        enforced: List[List[int]] = []
        cap = max_sentences_per_chunk
        for g in group_indices:
            if len(g) <= cap:
                enforced.append(g)
                continue
            # Split sequentially; allow last remainder < min_sentences_per_chunk if unavoidable
            for i in range(0, len(g), cap):
                sub = g[i:i+cap]
                if sub:
                    enforced.append(sub)
        group_indices = enforced
    
    del sim_matrix
    gc.collect()

    if not group_indices:
        if not silent:
            log_msg(silent, "No groups formed fallback original", 'warning', 'group')
        return [(f"{doc_id}_no_groups", passage_text, None)]

    # --- Final enforcement: merge sub-min chunks, split over-max chunks ---
    # Convert group_indices to list of sentence lists
    chunk_lists = [ [idx for idx in group if 0 <= idx < len(sentences)] for group in group_indices if group ]
    min_s = min_sentences_per_chunk if min_sentences_per_chunk and min_sentences_per_chunk > 1 else None
    max_s = max_sentences_per_chunk if max_sentences_per_chunk and max_sentences_per_chunk > 0 else None
    # Strict enforcement: merge all sub-min chunks (ignore max), then re-split to [min,max]
    if min_s:
        # Merge all sub-min chunks forward
        merged_chunks = []
        buffer = []
        for ch in chunk_lists:
            if not buffer:
                buffer = list(ch)
            else:
                buffer.extend(ch)
            while buffer and len(buffer) >= min_s:
                merged_chunks.append(buffer[:min_s])
                buffer = buffer[min_s:]
        if buffer:
            # If leftover < min, append to last if possible
            if merged_chunks and len(buffer) < min_s:
                merged_chunks[-1].extend(buffer)
            else:
                merged_chunks.append(buffer)
        chunk_lists = merged_chunks
    # Now re-split any chunk > max_s
    if max_s:
        adjusted = []
        for ch in chunk_lists:
            if len(ch) <= max_s:
                adjusted.append(ch)
            else:
                for j in range(0, len(ch), max_s):
                    sub = ch[j:j+max_s]
                    adjusted.append(sub)
        chunk_lists = adjusted
    # Final safeguard: if last chunk < min and >1 chunk, merge with previous
    if min_s and len(chunk_lists) > 1 and len(chunk_lists[-1]) < min_s:
        chunk_lists[-2].extend(chunk_lists[-1])
        chunk_lists.pop()
    # Build final chunks
    final_chunks: List[Tuple[str, str, Optional[str]]] = []
    for i, valid_indices in enumerate(chunk_lists):
        if not valid_indices:
            continue
        chunk_sentences = [sentences[idx] for idx in sorted(valid_indices)]
        chunk_text = " ".join(chunk_sentences).strip()
        if not chunk_text:
            continue
        chunk_hash = hashlib.sha1(chunk_text.encode('utf-8', errors='ignore')).hexdigest()[:8]
        chunk_id = f"{doc_id}_group{i}_hash{chunk_hash}"
        oie_string = None
        num_sentences = len(valid_indices)
        final_chunks.append((chunk_id, chunk_text, oie_string))
        if not silent:
            log_msg(silent, f"Chunk {i} preserved sentences={num_sentences}", 'debug', 'group')
    if not final_chunks:
        if not silent:
            log_msg(silent, "No valid chunks after filtering fallback", 'warning', 'group')
        final_chunks = [(f"{doc_id}_filter_fail", passage_text, None)]
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
    # (Character size control removed)
    silent: bool = False,
    device: Optional[str] = "cuda",
    target_sentences_per_chunk: Optional[int] = None,
    enforce_sentence_balancing: bool = True,
    sentence_target_tolerance: float = 0.5,
    max_sentences_per_chunk: Optional[int] = None,
    **kwargs
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Compatibility wrapper for the optimized semantic grouping.
    This function maintains the same interface as the original semantic_chunk_passage_from_grouping_logic
    but uses the optimized implementation.
    
    New Parameters:
        
    (Character size parameters removed)
        
    """
    
    # Ignore deprecated size-based kwargs if provided
    deprecated_keys = {k for k in list(kwargs.keys()) if 'chunk_size' in k or 'tolerance' in k}
    for k in deprecated_keys:
        kwargs.pop(k, None)
        if not silent:
            log_msg(False, f"Deprecated param '{k}' ignored (character size control removed)", 'warning', 'group')

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
        target_sentences_per_chunk=target_sentences_per_chunk,
        enforce_sentence_balancing=enforce_sentence_balancing,
        sentence_target_tolerance=sentence_target_tolerance,
    max_sentences_per_chunk=max_sentences_per_chunk,
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

# ================= Sentence-only grouping (no char control) =================
def sentence_count_grouping(
    sim_matrix: np.ndarray,
    sentences: List[str],
    target_sentences: int,
    min_sentences: int,
    tolerance: float,
    upper_threshold: float,
    lower_threshold: float,
    decay_factor: float,
    silent: bool = True,
) -> List[List[int]]:
    """Group sentences using only sentence counts and semantic similarity thresholds.

    Strategy:
      - Define allowed size window: L = max(min_sentences, target*(1 - tol)); U = target*(1 + tol)
      - Iterate sentences sequentially building groups.
      - Start a new group when:
          * current size >= U, OR
          * similarity with next sentence below lower_threshold AND current size >= L
      - Prevent tiny trailing group: if final group < min_sentences merge into previous.
    """
    if sim_matrix is None or len(sentences) == 0:
        return []
    total = len(sentences)
    L = max(min_sentences, int(max(1, round(target_sentences * (1 - tolerance)))))
    U = max(L, int(round(target_sentences * (1 + tolerance))))

    groups: List[List[int]] = []
    current: List[int] = []

    def flush(force: bool = False):
        nonlocal current
        if not current:
            return
        if (len(current) >= L) or force:
            groups.append(current)
            current = []

    for idx in range(total):
        current.append(idx)
        # Decide if we should close the group
        at_end = (idx == total - 1)
        cur_len = len(current)
        if at_end:
            flush(force=True)
            break

        if cur_len >= U:
            flush()
            continue

        # Similarity with next sentence
        next_idx = idx + 1
        sim_val = float(sim_matrix[idx, next_idx]) if sim_matrix is not None else 1.0
        # If similarity low and we already hit lower bound L -> cut
        if sim_val < lower_threshold and cur_len >= L:
            flush()
            continue

    # Post-process: ensure no last tiny group
    if len(groups) >= 2 and len(groups[-1]) < min_sentences:
        groups[-2].extend(groups[-1])
        groups.pop()

    if not silent:
        sizes = [len(g) for g in groups]
        if sizes:
            log_msg(False, f"Sentence-count grouping groups={len(groups)} size_avg={sum(sizes)/len(sizes):.2f} range={min(sizes)}-{max(sizes)} L={L} U={U}", 'info', 'group')
    return groups

def merge_small_groups_toward_sentence_target(
    groups: List[List[int]],
    target_sentences: int,
    sentence_tol: float,
    min_sentences: int,
    silent: bool
) -> List[List[int]]:
    """Merge consecutive small groups using ONLY sentence counts.

    Strategy:
      - Define window L..U around target (±tolerance)
      - Accumulate groups until size within window, then emit
      - Always preserve order; no character-size checks.
    """
    if not groups:
        return groups
    L = max(min_sentences, int(max(1, round(target_sentences * (1 - sentence_tol)))))
    U = max(L, int(round(target_sentences * (1 + sentence_tol))))
    merged: List[List[int]] = []
    buffer: List[int] = []

    def flush():
        nonlocal buffer
        if buffer:
            merged.append(sorted(buffer))
            buffer = []

    for g in sorted(groups, key=lambda g: min(g) if g else 10**9):
        if not g:
            continue
        g_len = len(g)
        if L <= g_len <= U:
            flush()
            merged.append(sorted(g))
            continue
        if not buffer:
            buffer = list(g)
        else:
            buffer.extend(g)
        # evaluate buffer
        while buffer:
            b_len = len(buffer)
            if b_len > U:
                if b_len >= L:
                    flush()
                break
            if b_len >= L:
                flush()
                break
            break
    flush()
    if len(merged) >= 2 and len(merged[-1]) < min_sentences:
        merged[-2].extend(merged[-1])
        merged.pop()
    if not silent:
        sizes = [len(g) for g in merged]
        if sizes:
            log_msg(False, f"Sentence-target merge groups={len(merged)} size_avg={sum(sizes)/len(sizes):.2f} range={min(sizes)}-{max(sizes)}", 'info', 'group')
    return merged
