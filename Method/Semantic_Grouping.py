import pandas as pd
from dotenv import load_dotenv
from Tool.Sentence_Detector import extract_and_simplify_sentences
from Tool.Sentence_Embedding import sentence_embedding as embed_text_list 
from Tool.OIE import extract_relations_from_paragraph 
import os 
import json 
import time 
import re 
import nltk 
import pickle 
from psycopg2 import extras as psycopg2_extras 
from typing import List, Tuple, Dict, Union, Optional 
import gc 
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import hashlib 
from datetime import datetime
from pathlib import Path

load_dotenv()

def count_tokens_simple(text: str) -> int:
    """Simple token counting using whitespace splitting"""
    if not text or not isinstance(text, str):
        return 0
    tokens = text.strip().split()
    return len(tokens)

def count_tokens_accurate(text: str) -> int:
    """More accurate token counting using regex patterns"""
    if not text or not isinstance(text, str):
        return 0
    import re
    # Split on word boundaries and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.strip())
    return len(tokens)

# --- Helper function for batched embedding ---
def embed_sentences_in_batches(sentences, model_name, batch_size=32, device=None, silent=True):
    """
    Embeds a list of sentences in batches using a specified sentence transformer model.
    Optimized for better memory management and error handling.
    """
    if not sentences:
        if not silent:
            print("No sentences provided to embed.")
        return np.array([])

    all_embeddings = []
    if not silent:
        print(f"Embedding {len(sentences)} sentences in batches of {batch_size}...")

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        if not silent and (i // batch_size) % 5 == 0:  # Progress every 5 batches
            print(f"  Processing batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}...")
        
        try:
            # Ensure device_preference is always str for typing
            device_pref = str(device) if device is not None else "cpu"
            batch_embeddings = embed_text_list(batch, model_name=model_name, batch_size=len(batch), device_preference=device_pref)
            if batch_embeddings is not None and len(batch_embeddings) > 0:
                all_embeddings.append(batch_embeddings)
            else:
                if not silent:
                    print(f"Warning: Batch {i // batch_size + 1} returned no embeddings.")
        except Exception as e:
            if not silent:
                print(f"Error during batched embedding batch {i // batch_size + 1}: {e}")
            # Continue with next batch instead of failing completely

    if not all_embeddings:
        if not silent:
            print("No embeddings were generated successfully.")
        return np.array([])

    try:
        # Concatenate all batch embeddings
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings
    except ValueError as e:
        if not silent:
            print(f"Error concatenating batch embeddings: {e}")
        return np.array([])

def create_semantic_matrix(sentences: List[str], model_name: str, batch_size_embed: int = 32, device: Optional[str] = None, silent: bool = True) -> Optional[np.ndarray]:
    """
    Create semantic similarity matrix between sentences using batched embedding.
    Optimized for memory efficiency.
    """
    if len(sentences) < 2:
        return None

    # Embed sentences in batches
    embeddings = embed_sentences_in_batches(
        sentences, 
        model_name=model_name, 
        batch_size=batch_size_embed, 
        device=device, 
        silent=silent
    )

    if embeddings is None or embeddings.shape[0] != len(sentences):
        if not silent:
            print("Error: Embedding failed or mismatch in number of embeddings.")
        # Clean up memory
        if 'embeddings' in locals() and embeddings is not None: 
            del embeddings
        gc.collect()
        return None

    try:
        # Calculate cosine similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        # Clean up embeddings after similarity calculation
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
    """
    Analyze similarity distribution in matrix, excluding diagonal values.
    Returns statistics for auto-threshold calculation.
    """
    if not isinstance(sim_matrix, np.ndarray) or sim_matrix.ndim != 2 or sim_matrix.shape[0] < 2:
        return None

    # Get upper triangle values (excluding diagonal)
    upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[upper_triangle_indices]

    # Filter out values very close to 1.0 (likely exact matches)
    epsilon = 1e-5
    filtered_similarities = similarities[similarities < (1.0 - epsilon)]

    if filtered_similarities.size == 0:
        if similarities.size > 0:
            # All values are 1.0
            original_max = np.max(similarities)
            return {
                'min': original_max, 'max': original_max, 'mean': original_max, 'std': 0.0,
                **{f'p{p}': original_max for p in [10, 25, 50, 75, 80, 85, 90, 95]}
            }
        return None

    # Calculate statistics on filtered values
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

def semantic_spreading_grouping_optimized(sim_matrix, sentences, initial_threshold, decay_factor, min_threshold, 
                                        min_chunk_len, max_chunk_len, silent=True):

    min_chunk_len = 0
    max_chunk_len = float('inf')
    num_sentences = sim_matrix.shape[0]
    ungrouped_indices = set(range(num_sentences))  # Use set for faster operations
    groups = []
    current_threshold = initial_threshold
    iteration = 0
    max_iterations = num_sentences * 2  # Prevent infinite loops

    if not silent:
        print(f"Starting semantic grouping with {num_sentences} sentences")
        print(f"Token constraints: {min_chunk_len}-{max_chunk_len}")
        print(f"Thresholds: initial={initial_threshold:.3f}, decay={decay_factor}, min={min_threshold:.3f}")

    while len(ungrouped_indices) > 0 and iteration < max_iterations:
        iteration += 1
        
        # Find valid starting pairs
        best_pairs = find_best_similarity_pairs(
            sim_matrix, list(ungrouped_indices), sentences, 
            current_threshold, max_chunk_len, silent
        )
        
        if not best_pairs:
            # Decay threshold and try again
            new_threshold = max(min_threshold, current_threshold * decay_factor)
            if new_threshold == current_threshold:  # Can't decay further
                break
            current_threshold = new_threshold
            if not silent:
                print(f"  No valid pairs found. Decaying threshold to {current_threshold:.4f}")
            continue

        # Process best pair to create new group
        best_pair, similarity_score = best_pairs[0]
        idx1, idx2 = best_pair
        
        # Initialize new group
        current_group = [idx1, idx2]
        ungrouped_indices.discard(idx1)
        ungrouped_indices.discard(idx2)
        
        # Expand group iteratively
        current_group = expand_group_with_constraints(
            sim_matrix, sentences, current_group, list(ungrouped_indices),
            current_threshold, max_chunk_len, silent
        )
        
        # Remove added sentences from ungrouped
        for idx in current_group:
            ungrouped_indices.discard(idx)
        
        # Validate group meets minimum token requirement
        group_text = " ".join([sentences[idx] for idx in current_group])
        group_tokens = count_tokens_accurate(group_text)
        
        if group_tokens >= min_chunk_len:
            groups.append(sorted(current_group))
            if not silent:
                print(f"  Group {len(groups)}: {len(current_group)} sentences, {group_tokens} tokens ✓")
        else:
            if not silent:
                print(f"  Group candidate: {group_tokens} tokens < {min_chunk_len}, discarding")
            # Put sentences back if they don't form valid group
            ungrouped_indices.update(current_group)

        # Apply threshold decay for next iteration
        current_threshold = max(min_threshold, initial_threshold * (decay_factor ** len(groups)))

    # Handle remaining individual sentences
    if ungrouped_indices:
        remaining_groups = handle_remaining_sentences(
            list(ungrouped_indices), sentences, min_chunk_len, max_chunk_len, silent
        )
        groups.extend(remaining_groups)

    if not silent:
        print(f"Semantic grouping completed: {len(groups)} groups formed")

    return groups

def find_best_similarity_pairs(sim_matrix, ungrouped_indices, sentences, threshold, max_tokens, silent):
    """Find pairs of sentences that meet similarity and token constraints"""
    valid_pairs = []
    
    for i, idx1 in enumerate(ungrouped_indices):
        for j in range(i + 1, len(ungrouped_indices)):
            idx2 = ungrouped_indices[j]
            similarity = sim_matrix[idx1, idx2]
            
            if similarity >= threshold:
                # Check token constraint
                combined_text = sentences[idx1] + " " + sentences[idx2]
                combined_tokens = count_tokens_accurate(combined_text)
                
                if combined_tokens <= max_tokens:
                    valid_pairs.append(((idx1, idx2), similarity))

    # Sort by similarity score descending
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    return valid_pairs

def expand_group_with_constraints(sim_matrix, sentences, current_group, ungrouped_indices, 
                                threshold, max_tokens, silent):
    """Expand a group by adding similar sentences while respecting token constraints"""
    added_in_iteration = True
    max_expansion_iterations = len(ungrouped_indices) + 1
    expansion_iteration = 0
    
    while added_in_iteration and expansion_iteration < max_expansion_iterations:
        expansion_iteration += 1
        added_in_iteration = False
        candidates_to_add = []
        
        for ungrouped_idx in list(ungrouped_indices):  # Create copy for safe iteration
            # Check similarity with any member of current group
            max_similarity = -1
            for group_member_idx in current_group:
                similarity = sim_matrix[ungrouped_idx, group_member_idx]
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= threshold:
                # Check token constraint
                test_group_sentences = [sentences[idx] for idx in current_group + [ungrouped_idx]]
                test_text = " ".join(test_group_sentences)
                test_tokens = count_tokens_accurate(test_text)
                
                if test_tokens <= max_tokens:
                    candidates_to_add.append((ungrouped_idx, max_similarity))

        # Add candidates sorted by similarity (best first)
        if candidates_to_add:
            candidates_to_add.sort(key=lambda x: x[1], reverse=True)
            for candidate_idx, similarity in candidates_to_add:
                if candidate_idx in ungrouped_indices:  # Double-check it's still available
                    current_group.append(candidate_idx)
                    ungrouped_indices.remove(candidate_idx)
                    added_in_iteration = True
                    if not silent:
                        print(f"    Added sentence {candidate_idx} (similarity: {similarity:.3f})")

    return current_group

def handle_remaining_sentences(remaining_indices, sentences, min_tokens, max_tokens, silent):
    """Handle sentences that couldn't be grouped, trying to merge small ones"""
    if not remaining_indices:
        return []
    
    groups = []
    current_merge_group = []
    current_merge_tokens = 0
    
    if not silent:
        print(f"Handling {len(remaining_indices)} remaining sentences...")
    
    for idx in remaining_indices:
        sentence_tokens = count_tokens_accurate(sentences[idx])
        
        if sentence_tokens >= min_tokens:
            # Sentence is large enough on its own
            if sentence_tokens <= max_tokens:
                groups.append([idx])
                if not silent:
                    print(f"  Individual sentence {idx}: {sentence_tokens} tokens ✓")
            else:
                # Sentence is too large, try to split it
                split_groups = split_large_sentence(sentences[idx], idx, min_tokens, max_tokens)
                groups.extend(split_groups)
        else:
            # Try to merge with other small sentences
            test_tokens = current_merge_tokens + sentence_tokens
            if test_tokens <= max_tokens:
                current_merge_group.append(idx)
                current_merge_tokens = test_tokens
            else:
                # Save current merge group if valid
                if current_merge_tokens >= min_tokens and current_merge_group:
                    groups.append(current_merge_group)
                    if not silent:
                        print(f"  Merged group: {len(current_merge_group)} sentences, {current_merge_tokens} tokens ✓")
                
                # Start new merge group
                current_merge_group = [idx]
                current_merge_tokens = sentence_tokens
    
    # Handle final merge group
    if current_merge_group:
        if current_merge_tokens >= min_tokens:
            groups.append(current_merge_group)
            if not silent:
                print(f"  Final merged group: {len(current_merge_group)} sentences, {current_merge_tokens} tokens ✓")
        else:
            if not silent:
                print(f"  Discarding final merge group: {current_merge_tokens} < {min_tokens} tokens")
    
    return groups

def split_large_sentence(sentence_text, sentence_idx, min_tokens, max_tokens):
    """Split a sentence that's too large into smaller chunks"""
    # Simple approach: split by punctuation or at word boundaries
    words = sentence_text.split()
    if len(words) <= 1:
        return []  # Can't split further
    
    chunks = []
    current_chunk_words = []
    current_tokens = 0
    
    for word in words:
        word_tokens = count_tokens_accurate(word)
        test_tokens = current_tokens + word_tokens
        
        if test_tokens <= max_tokens:
            current_chunk_words.append(word)
            current_tokens = test_tokens
        else:
            # Save current chunk if it meets minimum
            if current_tokens >= min_tokens:
                chunk_text = " ".join(current_chunk_words)
                chunks.append([f"{sentence_idx}_split_{len(chunks)}"])
            
            # Start new chunk
            current_chunk_words = [word]
            current_tokens = word_tokens
    
    # Handle final chunk
    if current_tokens >= min_tokens:
        chunks.append([f"{sentence_idx}_split_{len(chunks)}"])
    
    return chunks

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
        
        if s and r and o:  # Only include if all parts are non-empty
            formatted_sentences.append(f"{s} {r} {o}.")
    
    if not formatted_sentences:
        return None

    return " ".join(formatted_sentences).strip()

def _force_split_too_large_chunk(text: str, max_tokens: int) -> List[str]:
    """
    Splits a chunk of text that is over max_tokens into smaller chunks,
    respecting sentence boundaries.
    """
    sentences = extract_and_simplify_sentences(text, simplify=False)
    if not sentences:
        # Fallback for text without clear sentences
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    chunks = []
    current_chunk_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_len = count_tokens_accurate(sentence)

        # If a single sentence is larger than max_tokens, split it by words
        if sentence_len > max_tokens:
            if current_chunk_sentences: # Finalize previous chunk first
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0
            
            words = sentence.split()
            sub_chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
            chunks.extend(sub_chunks)
            continue

        # If adding the next sentence exceeds the limit, finalize the current chunk
        if current_chunk_sentences and current_tokens + sentence_len > max_tokens:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_tokens = sentence_len
        else:
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_len
    
    # Add the last remaining chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return [chunk for chunk in chunks if chunk]

def extract_oie_for_chunk(chunk_text: str, max_triples: Optional[int] = None, silent: bool = True) -> Optional[str]:
    """Extract OIE relations for a chunk and format them"""
    if not chunk_text or not chunk_text.strip():
        return None
    
    try:
        if not silent:
            print(f"    Extracting OIE for chunk ({count_tokens_accurate(chunk_text)} tokens)...")
        
        # Trích xuất các mối quan hệ từ chunk bằng OIE
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

def semantic_chunk_passage_from_grouping_logic(
    passage_text: str,
    doc_id: str,
    embedding_model: str,  
    initial_threshold: Union[str, float] = "auto",
    decay_factor: float = 0.85,
    min_threshold: Union[str, float] = "auto",
    min_chunk_len_tokens: int = 50,      
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

    # --- VÔ HIỆU HÓA RÀNG BUỘC MIN TOKEN ---
    min_chunk_len_tokens = 0

    if not silent:
        print(f"🔄 Processing passage {doc_id} with Enhanced Semantic Grouping")
        print(f"   Model: {embedding_model}")
        print(f"   Token range: {min_chunk_len_tokens}-{max_chunk_len_tokens}")
        print(f"   OIE enabled: {include_oie}")

    # 1. Extract sentences
    sentences = extract_and_simplify_sentences(passage_text, simplify=False)
    if not sentences:
        if not silent:
            print(f"   No sentences found in passage {doc_id}")
        
        # Return original passage with OIE if requested
        oie_string = None
        final_text = passage_text
        if include_oie:
            print(f"🔧 DEBUG: OIE enabled for semantic_grouping")
            oie_string = extract_oie_for_chunk(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        return [(f"{doc_id}_single_empty", final_text, oie_string)]

    # 2. Check if passage is small enough to keep as single chunk
    total_tokens = count_tokens_accurate(passage_text)
    if total_tokens <= max_chunk_len_tokens:
        if not silent:
            print(f"   Passage fits in single chunk ({total_tokens} tokens)")
        
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = extract_oie_for_chunk(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        return [(f"{doc_id}_single_complete", final_text, oie_string)]

    # 3. Create similarity matrix
    if not silent:
        print(f"   Creating similarity matrix for {len(sentences)} sentences...")
    
    sim_matrix = create_semantic_matrix(
        sentences, 
        model_name=embedding_model, 
        batch_size_embed=embedding_batch_size, 
        device=kwargs.get('device'), 
        silent=silent
    )
    
    if sim_matrix is None:
        if not silent:
            print(f"   Failed to create similarity matrix")
        
        # Fallback to original passage
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = extract_oie_for_chunk(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        return [(f"{doc_id}_matrix_fail", final_text, oie_string)]

    # 4. Resolve auto thresholds
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
            
            # Ensure numeric comparison
            try:
                init_val = float(actual_initial_threshold)
                min_val  = float(actual_min_threshold)
            except (TypeError, ValueError):
                init_val, min_val = 0.75, 0.5  # sensible defaults

            # Ensure initial >= min
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

    # 5. Perform semantic grouping
    if not silent:
        print(f"   Grouping with thresholds: {actual_initial_threshold:.4f} → {actual_min_threshold:.4f}")
    
    group_indices = semantic_spreading_grouping_optimized(
        sim_matrix, sentences, actual_initial_threshold, decay_factor, 
        actual_min_threshold, min_chunk_len_tokens, max_chunk_len_tokens, silent
    )
    
    # Clean up similarity matrix
    del sim_matrix
    gc.collect()

    if not group_indices:
        if not silent:
            print(f"   No groups formed, returning original passage")
        
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = extract_oie_for_chunk(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        return [(f"{doc_id}_no_groups", final_text, oie_string)]

    # 6. Create final chunks with OIE
    if not silent:
        print(f"   Creating {len(group_indices)} chunks with OIE processing...")
    
    final_chunks = []
    all_raw_oie_data: List[Dict] = []  # Thu thập OIE thô để ghi ra file
    
    for i, group_sentence_indices in enumerate(group_indices):
        # Validate indices
        valid_indices = [idx for idx in group_sentence_indices if 0 <= idx < len(sentences)]
        if not valid_indices:
            continue
        
        # Create chunk text
        chunk_sentences = [sentences[idx] for idx in sorted(valid_indices)]
        chunk_text = " ".join(chunk_sentences).strip()
        
        if not chunk_text:
            continue
        
        # Generate chunk ID
        chunk_hash = hashlib.sha1(chunk_text.encode('utf-8', errors='ignore')).hexdigest()[:8]
        chunk_id = f"{doc_id}_group{i}_hash{chunk_hash}"
        
        # Extract OIE if requested (lưu cả raw)
        oie_string = None
        final_chunk_text = chunk_text
        raw_oie_relations = None

        if include_oie and chunk_text.strip():
            try:
                raw_oie_relations = extract_relations_from_paragraph(chunk_text, silent=True)  # type: ignore[arg-type]
                if raw_oie_relations:
                    oie_string = format_oie_triples_to_string(raw_oie_relations)
                    if oie_string:
                        final_chunk_text = f"{chunk_text} {oie_string}"

                    # Thu thập để lưu nếu cần
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
        
        # Validate final token count based on the original chunk text
        base_tokens = count_tokens_accurate(chunk_text)
        if base_tokens >= min_chunk_len_tokens:
            final_chunks.append((chunk_id, final_chunk_text, oie_string))
            if not silent:
                final_tokens = count_tokens_accurate(final_chunk_text)
                print(f"   ✓ Chunk {i}: {base_tokens} base tokens, {final_tokens} final tokens")
        else:
            if not silent:
                print(f"   ✗ Chunk {i}: {base_tokens} base tokens < {min_chunk_len_tokens} (discarded)")

    # 7. Handle case where no valid chunks were created
    if not final_chunks:
        if not silent:
            print(f"   No valid chunks after filtering, returning original passage")
        
        oie_string = None
        final_text = passage_text
        if include_oie:
            oie_string = extract_oie_for_chunk(passage_text, silent=silent)
            if oie_string:
                final_text = f"{passage_text} {oie_string}"
        
        final_chunks = [(f"{doc_id}_filter_fail", final_text, oie_string)]

    # 8. Force-split any chunks that are still too large to meet ideal training size
    if not silent:
        print(f"   Ensuring all chunks respect max token limit ({max_chunk_len_tokens})...")

    force_split_chunks = []
    for chunk_id, chunk_text_w_oie, oie_string_from_chunker in final_chunks:
        # Check size of the base text (before OIE was added)
        base_text = chunk_text_w_oie
        if oie_string_from_chunker and base_text.endswith(oie_string_from_chunker):
             # Strip the OIE part to get the base text for accurate size check
             base_text = base_text[:-len(oie_string_from_chunker)].strip()
        
        base_token_count = count_tokens_accurate(base_text)

        if base_token_count > max_chunk_len_tokens:
            if not silent:
                print(f"     Chunk {chunk_id} ({base_token_count} tokens) is too large. Force-splitting...")
            
            sub_chunks_text = _force_split_too_large_chunk(base_text, max_chunk_len_tokens)
            
            for i, sub_chunk_text_item in enumerate(sub_chunks_text):
                sub_chunk_id = f"{chunk_id}_fs{i}" # fs for force_split
                
                # Re-apply OIE to the new, smaller sub-chunks
                sub_oie_string = None
                final_sub_chunk_text = sub_chunk_text_item
                if include_oie:
                    sub_oie_string = extract_oie_for_chunk(sub_chunk_text_item, silent=silent)
                    if sub_oie_string:
                        final_sub_chunk_text = f"{sub_chunk_text_item} {sub_oie_string}"
                
                force_split_chunks.append((sub_chunk_id, final_sub_chunk_text, sub_oie_string))
        else:
            # Chunk is already a good size
            force_split_chunks.append((chunk_id, chunk_text_w_oie, oie_string_from_chunker))

    final_chunks = force_split_chunks

    # 9. Lưu raw OIE nếu cần
    if save_raw_oie and all_raw_oie_data:
        try:
            raw_oie_filepath = save_raw_oie_data(all_raw_oie_data, doc_id, output_dir, "semantic_grouping")
            if raw_oie_filepath and not silent:
                print(f"📄 Raw OIE data saved to: {raw_oie_filepath}")
                total_relations = sum(entry['relation_count'] for entry in all_raw_oie_data)
                print(f"   Total chunks with OIE: {len(all_raw_oie_data)}")
                print(f"   Total relations extracted: {total_relations}")
        except Exception as e:
            if not silent:
                print(f"Warning: Failed to save raw OIE data: {e}")

    if not silent:
        token_counts = [count_tokens_accurate(chunk[1]) for chunk in final_chunks]
        if token_counts:
            print(f"   ✅ Created {len(final_chunks)} final chunks after force-splitting.")
            print(f"   📊 Final token distribution: {min(token_counts)}-{max(token_counts)} (avg: {sum(token_counts)/len(token_counts):.1f})")
        else:
            print("   No chunks were created after processing.")

    return final_chunks

# Alias for backward compatibility
semantic_spreading_grouping_with_token_control = semantic_spreading_grouping_optimized

# -------------------------------------------------------------
# Helper: Lưu raw OIE data (sao chép từ Text_Splitter)
# -------------------------------------------------------------

def save_raw_oie_data(oie_data: List[Dict], chunk_id: str, output_dir: str, method_name: str = "semantic_grouping") -> Optional[str]:
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