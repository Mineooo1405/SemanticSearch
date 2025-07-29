import json
from typing import List, Dict, Union, Optional, Tuple
from dotenv import load_dotenv
from Tool.Sentence_Segmenter import extract_sentences_spacy, count_tokens_spacy
from Tool.OIE import extract_relations_from_paragraph
import numpy as np
from Tool.Sentence_Embedding import sentence_embedding as embed_text_list
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import traceback
from datetime import datetime
from pathlib import Path

load_dotenv()

def _count_tokens_accurate(text: str) -> int:
    """Token counting using spaCy tokenizer"""
    return count_tokens_spacy(text)

def process_sentence_splitting_with_semantics(
    text: str, 
    chunk_size: Optional[int] = None,  # None = không giới hạn kích thước dựa ký tự
    chunk_overlap: int = 0,
    target_tokens: Optional[int] = None,
    tolerance: float = 0.25,
    enable_adaptive: bool = False,
    semantic_threshold: Optional[float] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    device: Optional[object] = None,
) -> Tuple[List[str], List[str], List[List[int]]]:
    """
    Optimized sentence-based text splitting with adaptive token control.
    
    Returns:
        - chunks: List of chunk texts
        - sentences: List of original sentences
        - sentence_groups: List of sentence indices for each chunk
    """
    
    # Extract sentences
    sentences = extract_sentences_spacy(text)

    # Compute embeddings when semantic_threshold is set
    embeddings: Optional[np.ndarray] = None
    if semantic_threshold is not None and sentences:
        try:
            device_pref = str(device) if device is not None else "cpu"
            embeddings = np.array(
                embed_text_list(
                    sentences,
                    model_name=embedding_model,
                    batch_size=32,
                    device_preference=device_pref,
                )
            )
            # Note: cosine_similarity handles normalization internally, 
            # but pre-normalization can still improve numerical stability
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            embeddings = embeddings / norms
        except Exception as e:
            print(f"Semantic embedding failed ({e}); fallback to length-only splitter")
            semantic_threshold = None
            embeddings = None
    
    if not sentences:
        return [text] if text.strip() else [], [], [[0]] if text.strip() else []
    
    # Nếu không đặt giới hạn chunk_size, bỏ qua check “small enough”.
    if chunk_size is not None and len(text) <= chunk_size:
        sentence_indices = list(range(len(sentences)))
        return [text], sentences, [sentence_indices]
    
    # Adaptive chunking parameters – vẫn hoạt động nếu chunk_size None (cho phép chia tự nhiên)
    if enable_adaptive and target_tokens and chunk_size is not None:
        # Convert target tokens to approximate character count
        # Rough estimate: 1 token ≈ 4.5 characters (including spaces)
        target_chars = int(target_tokens * 4.5)
        min_chars = int(target_chars * (1 - tolerance))
        max_chars = int(target_chars * (1 + tolerance))
        
        # Use adaptive targets instead of fixed chunk_size
        chunk_size = target_chars
    
    # Calculate sentence lengths
    sentence_lengths = [len(s) for s in sentences]
    
    chunks = []
    current_chunk_sentences = []
    current_size = 0
    sentence_groups = []
    current_group_indices = []
    
    # Nếu chunk_size None, đặt thành float('inf') để bỏ qua giới hạn độ dài
    effective_chunk_size = float('inf') if chunk_size is None else chunk_size

    for i, sentence in enumerate(sentences):
        sentence_len = sentence_lengths[i]
        
        # Semantic cut based on similarity with previous sentence
        if (
            semantic_threshold is not None
            and current_chunk_sentences
            and embeddings is not None
        ):
            last_idx = current_group_indices[-1]
            sim_val = float(cosine_similarity(
                embeddings[last_idx].reshape(1, -1), 
                embeddings[i].reshape(1, -1)
            )[0, 0])
            if sim_val < semantic_threshold:
                # Finish current chunk before adding new sentence
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(chunk_text)
                sentence_groups.append(list(current_group_indices))

                # Handle overlap if requested
                if chunk_overlap > 0 and len(current_chunk_sentences) > 1:
                    overlap_sentences, overlap_indices = helper_create_overlap(
                        current_chunk_sentences, current_group_indices, chunk_overlap
                    )
                    current_chunk_sentences = overlap_sentences
                    current_group_indices = overlap_indices
                    current_size = sum(len(s) for s in current_chunk_sentences)
                    if len(current_chunk_sentences) > 1:
                        current_size += len(current_chunk_sentences) - 1
                else:
                    current_chunk_sentences = []
                    current_group_indices = []
                    current_size = 0
        
        # Handle sentences that are too long for any chunk
        if sentence_len > effective_chunk_size:
            # Save current chunk if exists
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(chunk_text)
                sentence_groups.append(list(current_group_indices))
                current_chunk_sentences = []
                current_group_indices = []
                current_size = 0
            
            # Split long sentence if adaptive mode is enabled
            if enable_adaptive and target_tokens:
                sub_chunks = helper_split_long_sentence(sentence, target_chars, max_chars)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append(sub_chunk)
                    sentence_groups.append([f"{i}_split_{j}"])
            else:
                # Keep as single chunk (original behavior)
                chunks.append(sentence)
                sentence_groups.append([i])
            continue
        
        # Check if adding this sentence would exceed chunk size
        projected_size = current_size + sentence_len
        if current_chunk_sentences:
            projected_size += 1  # Space between sentences
        
        if projected_size > effective_chunk_size and current_chunk_sentences:
            # Save current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
            sentence_groups.append(list(current_group_indices))
            
            # Handle overlap (create overlap from end of current chunk)
            if chunk_overlap > 0 and len(current_chunk_sentences) > 1:
                overlap_sentences, overlap_indices = helper_create_overlap(
                    current_chunk_sentences, current_group_indices, chunk_overlap
                )
                current_chunk_sentences = overlap_sentences
                current_group_indices = overlap_indices
                current_size = sum(len(s) for s in current_chunk_sentences)
                if len(current_chunk_sentences) > 1:
                    current_size += len(current_chunk_sentences) - 1  # Spaces
            else:
                # No overlap - start fresh
                current_chunk_sentences = []
                current_group_indices = []
                current_size = 0
        
        # Add current sentence to chunk
        current_chunk_sentences.append(sentence)
        current_group_indices.append(i)
        current_size = sum(len(s) for s in current_chunk_sentences)
        if len(current_chunk_sentences) > 1:
            current_size += len(current_chunk_sentences) - 1  # Spaces
    
    # Add final chunk if exists
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(chunk_text)
        sentence_groups.append(list(current_group_indices))
    
    return chunks, sentences, sentence_groups

def helper_create_overlap(
    current_sentences: List[str], 
    current_indices: List[int], 
    overlap_size: int
) -> Tuple[List[str], List[int]]:
    """Create overlap sentences from the end of current chunk"""
    if not current_sentences or overlap_size <= 0:
        return [], []
    
    overlap_sentences = []
    overlap_indices = []
    overlap_char_count = 0
    
    # Work backwards from end of current chunk
    for i in range(len(current_sentences) - 1, -1, -1):
        sentence = current_sentences[i]
        sentence_len = len(sentence)
        
        # Check if adding this sentence would exceed overlap size
        projected_overlap = overlap_char_count + sentence_len
        if len(overlap_sentences) > 0:
            projected_overlap += 1  # Space
        
        if projected_overlap > overlap_size and overlap_sentences:
            break
        
        overlap_sentences.insert(0, sentence)
        overlap_indices.insert(0, current_indices[i])
        overlap_char_count = projected_overlap
    
    return overlap_sentences, overlap_indices

def helper_split_long_sentence(sentence: str, target_chars: int, max_chars: int) -> List[str]:
    """Split a sentence that's too long into smaller parts"""
    if len(sentence) <= max_chars:
        return [sentence]
    
    # Try to split on punctuation first
    parts = []
    remaining = sentence
    
    while len(remaining) > max_chars:
        # Find good break points (punctuation, then spaces)
        break_point = helper_find_good_break_point(remaining, target_chars, max_chars)
        
        if break_point > 0:
            part = remaining[:break_point].strip()
            if part:
                parts.append(part)
            remaining = remaining[break_point:].strip()
        else:
            # Force break at max_chars if no good break point found
            part = remaining[:max_chars].strip()
            if part:
                parts.append(part)
            remaining = remaining[max_chars:].strip()
    
    # Add remaining part
    if remaining.strip():
        parts.append(remaining.strip())
    
    return parts

def helper_find_good_break_point(text: str, target: int, max_size: int) -> int:
    """Find a good break point in text around the target position"""
    if len(text) <= target:
        return len(text)
    
    # Look for punctuation near target
    search_start = max(0, target - 50)
    search_end = min(len(text), target + 50)
    
    # Priority break points
    punctuation_marks = ['. ', '! ', '? ', '; ', ', ']
    
    for punct in punctuation_marks:
        pos = text.rfind(punct, search_start, search_end)
        if pos > 0:
            return pos + len(punct)
    
    # Fall back to word boundaries
    pos = text.rfind(' ', search_start, min(max_size, len(text)))
    if pos > 0:
        return pos + 1
    
    # Last resort: break at max_size
    return min(max_size, len(text))

def transform_oie_triples_to_string(triples_list: List[Dict[str, str]], max_triples: Optional[int] = None) -> Optional[str]:
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

def process_oie_extraction(chunk_text: str, max_triples: Optional[int] = None, silent: bool = True) -> Optional[str]:
    """Extract OIE relations for a chunk and format them"""
    if not chunk_text or not chunk_text.strip():
        return None
    
    try:
        if not silent:
            print(f"    Extracting OIE for chunk ({_count_tokens_accurate(chunk_text)} tokens)...")
        
        relations = extract_relations_from_paragraph(chunk_text)
        
        if relations:
            oie_string = transform_oie_triples_to_string(relations, max_triples=max_triples)
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

def helper_save_raw_oie_data(oie_data: List[Dict], chunk_id: str, output_dir: str, method_name: str = "text_splitter") -> Optional[str]:
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
            "total_relations": len(oie_data)
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

def helper_merge_small_chunks(buffer: List[Tuple[str, str, Optional[str]]], context_id: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Merge small chunks in buffer"""
    if not buffer:
        return None
    
    if len(buffer) == 1:
        return buffer[0]
    
    # Merge texts
    merged_text = " ".join([chunk[1] for chunk in buffer])
    merged_id = f"{buffer[0][0]}_merged_{context_id}"
    
    # Merge available OIE strings (if any)
    merged_oie_parts = [chunk[2] for chunk in buffer if chunk[2]]
    merged_oie = " ".join(merged_oie_parts) if merged_oie_parts else None
    return (merged_id, merged_text, merged_oie)

def helper_split_large_chunk_by_tokens(text: str, chunk_id: str, max_tokens: int) -> List[Tuple[str, str, Optional[str]]]:
    """Split large chunks by token count using sentence boundaries"""
    sentences = extract_sentences_spacy(text)
    if len(sentences) <= 1:
        return [(chunk_id, text, None)]  # Can't split further
    
    sub_chunks = []
    current_sentences = []
    current_tokens = 0
    sub_idx = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = _count_tokens_accurate(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            # Create sub-chunk
            sub_text = ' '.join(current_sentences)
            sub_id = f"{chunk_id}_split_{sub_idx}"
            sub_chunks.append((sub_id, sub_text, None))
            
            current_sentences = [sentence]
            current_tokens = sentence_tokens
            sub_idx += 1
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
    
    # Handle final sentences
    if current_sentences:
        sub_text = ' '.join(current_sentences)
        sub_id = f"{chunk_id}_split_{sub_idx}"
        sub_chunks.append((sub_id, sub_text, None))
    
    return sub_chunks

def semantic_splitter_main(
    doc_id: str,
    passage_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    include_oie: bool = False,
    save_raw_oie: bool = False,
    output_dir: str = "./output",
    device: Optional[object] = None,
    target_tokens: int = 120,
    tolerance: float = 0.25,
    enable_adaptive: bool = True,
    semantic_threshold: float = 0.30,
    embedding_model: str = "all-MiniLM-L6-v2",
    silent: bool = False,
    **kwargs 
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Enhanced Text Splitter with adaptive token control and improved OIE integration.
    
    Returns:
        List of tuples: (chunk_id, chunk_text_with_oie, oie_string_only)
    """
    
    if not silent:
        print(f"    Processing passage {doc_id} with Enhanced Text Splitter")
        print(f"   Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars")
        print(f"   Target tokens: {target_tokens} ±{tolerance*100:.0f}%")
        print(f"   OIE enabled: {include_oie}")
        print(f"   Adaptive mode: {enable_adaptive}")

    try:
        # Check if passage is small enough to keep as single chunk
        total_tokens = _count_tokens_accurate(passage_text)
        
        if enable_adaptive and target_tokens:
            max_tokens = int(target_tokens * (1 + tolerance))
            if total_tokens <= max_tokens:
                if not silent:
                    print(f"   Passage fits in single chunk ({total_tokens} tokens)")
                
                # Process as single chunk with OIE
                oie_string = None
                final_text = passage_text
                
                if include_oie:
                    oie_string = process_oie_extraction(passage_text, silent=silent)
                    if oie_string:
                        final_text = f"{passage_text} {oie_string}"
                
                return [(f"{doc_id}_single_complete", final_text, oie_string)]
        
        # Split text into chunks using optimized algorithm
        if not silent:
            print(f"   Splitting into chunks...")
        
        chunk_texts, sentences, sentence_groups = process_sentence_splitting_with_semantics(
            text=passage_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            target_tokens=target_tokens,
            tolerance=tolerance,
            enable_adaptive=enable_adaptive,
            semantic_threshold=semantic_threshold,
            embedding_model=embedding_model,
            device=device,
        )
        
        if not chunk_texts:
            if not silent:
                print(f"   No chunks generated, returning original passage")
            
            oie_string = None
            final_text = passage_text
            if include_oie:
                oie_string = process_oie_extraction(passage_text, silent=silent)
                if oie_string:
                    final_text = f"{passage_text} {oie_string}"
            
            return [(f"{doc_id}_single_fallback", final_text, oie_string)]
        
        if not silent:
            print(f"   Generated {len(chunk_texts)} initial chunks")
        
        # Process chunks with OIE
        chunks_with_oie = []
        all_raw_oie_data = []
        
        for chunk_idx, chunk_text_content in enumerate(chunk_texts):
            chunk_id = f"{doc_id}_textsplit_{chunk_idx}"
            oie_string = None
            raw_oie_relations = None
            
            # Generate chunk hash for uniqueness
            chunk_hash = hashlib.sha1(chunk_text_content.encode('utf-8', errors='ignore')).hexdigest()[:8]
            chunk_id = f"{doc_id}_chunk{chunk_idx}_hash{chunk_hash}"
            
            # Extract OIE if requested
            final_chunk_text = chunk_text_content
            
            if include_oie and chunk_text_content.strip():
                try:
                    if not silent:
                        print(f"     Processing OIE for chunk {chunk_idx}...")
                    
                    raw_oie_relations = extract_relations_from_paragraph(chunk_text_content)
                    
                    if raw_oie_relations:
                        oie_string = transform_oie_triples_to_string(raw_oie_relations)
                        
                        if oie_string:
                            final_chunk_text = f"{chunk_text_content} {oie_string}"
                            if not silent:
                                print(f"       Added {len(raw_oie_relations)} OIE relations")
                        
                        # Save raw OIE data for analysis
                        if save_raw_oie:
                            all_raw_oie_data.append({
                                "chunk_id": chunk_id,
                                "relations": raw_oie_relations,
                                "relation_count": len(raw_oie_relations),
                                "chunk_text_preview": chunk_text_content[:100] + "..." if len(chunk_text_content) > 100 else chunk_text_content
                            })
                    else:
                        if not silent:
                            print(f"       No OIE relations found")
                            
                except Exception as e_oie:
                    if not silent:
                        print(f"       Error during OIE extraction: {e_oie}")
                    oie_string = None
            
            chunks_with_oie.append((chunk_id, final_chunk_text, oie_string))        
        
        # Save raw OIE data if requested
        if save_raw_oie and all_raw_oie_data:
            try:
                raw_oie_filepath = helper_save_raw_oie_data(all_raw_oie_data, doc_id, output_dir, "text_splitter")
                if raw_oie_filepath and not silent:
                    print(f" Raw OIE data saved to: {raw_oie_filepath}")
                    total_relations = sum(entry['relation_count'] for entry in all_raw_oie_data)
                    print(f"   Total chunks with OIE: {len(all_raw_oie_data)}")
                    print(f"   Total relations extracted: {total_relations}")
            except Exception as e:
                if not silent:
                    print(f"Warning: Failed to save raw OIE data: {e}")
        
        # Final validation and logging
        if not chunks_with_oie:
            if not silent:
                print(f"   No valid chunks after processing, returning original passage")
            
            oie_string = None
            final_text = passage_text
            if include_oie:
                oie_string = process_oie_extraction(passage_text, silent=silent)
                if oie_string:
                    final_text = f"{passage_text} {oie_string}"
            
            return [(f"{doc_id}_fallback_complete", final_text, oie_string)]
        
        # Log final statistics
        if not silent:
            token_counts = [_count_tokens_accurate(chunk[1]) for chunk in chunks_with_oie]
            print(f"   Created {len(chunks_with_oie)} chunks")
            print(f"   Token distribution: {min(token_counts)}-{max(token_counts)} (avg: {sum(token_counts)/len(token_counts):.1f})")
            
            if enable_adaptive and target_tokens:
                min_target = int(target_tokens * (1 - tolerance))
                max_target = int(target_tokens * (1 + tolerance))
                in_range = sum(1 for t in token_counts if min_target <= t <= max_target)
                compliance = in_range / len(token_counts) * 100
                print(f"   Target compliance: {in_range}/{len(token_counts)} ({compliance:.1f}%)")
        
        return chunks_with_oie

    except Exception as e:
        if not silent:
            print(f"Error chunking doc {doc_id} with Text Splitter: {e}")
            traceback.print_exc()
        
        # Return original passage as single chunk in case of error
        oie_string = None
        final_text = passage_text
        if include_oie:
            try:
                oie_string = process_oie_extraction(passage_text, silent=True)
                if oie_string:
                    final_text = f"{passage_text} {oie_string}"
            except:
                pass  # Ignore OIE errors in fallback
        
        return [(f"{doc_id}_error_fallback", final_text, oie_string)]

# Compatibility function to match the interface expected by data_create_controller
def chunk_passage_text_splitter(
    doc_id: str,
    passage_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    include_oie: bool = False,
    save_raw_oie: bool = False,
    output_dir: str = "./output",
    device: Optional[object] = None,
    target_tokens: int = 120,
    tolerance: float = 0.25,
    enable_adaptive: bool = True,
    semantic_threshold: float = 0.5,
    embedding_model: str = "all-MiniLM-L6-v2",
    silent: bool = False,
    **kwargs 
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Compatibility wrapper for the optimized semantic splitter.
    This function maintains the same interface as the original chunk_passage_text_splitter
    but uses the optimized implementation.
    """
    return semantic_splitter_main(
        doc_id=doc_id,
        passage_text=passage_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_oie=include_oie,
        save_raw_oie=save_raw_oie,
        output_dir=output_dir,
        device=device,
        target_tokens=target_tokens,
        tolerance=tolerance,
        enable_adaptive=enable_adaptive,
        semantic_threshold=semantic_threshold,
        embedding_model=embedding_model,
        silent=silent,
        **kwargs
    )
