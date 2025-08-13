import json  # (legacy import – may be removed if unused elsewhere)
import re
from typing import List, Dict, Union, Optional, Tuple
from dotenv import load_dotenv
from Tool.Sentence_Segmenter import extract_sentences_spacy, count_tokens_spacy
from .semantic_common import (
    normalize_device,
    estimate_optimal_batch_size,
    embed_sentences_batched,
    log_msg,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import traceback
import torch
from datetime import datetime
from pathlib import Path

load_dotenv()

def _count_tokens_accurate(text: str) -> int:
    """Token counting using spaCy tokenizer"""
    return count_tokens_spacy(text)

def _normalize_device(device):  # backward compat shim (no decorator to avoid call failure)
    return normalize_device(device)

def _estimate_optimal_batch_size(sentences, base_batch_size, device):  # backward compat shim
    return estimate_optimal_batch_size(sentences, base_batch_size, device)

def embed_sentences_in_batches(sentences: List[str], model_name: str, base_batch_size: int = 32, device: Optional[object] = None, silent: bool = True) -> Optional[np.ndarray]:
    return embed_sentences_batched(sentences, model_name, base_batch_size=base_batch_size, device=device, silent=silent)

def process_sentence_splitting_with_semantics(
    text: str,
    chunk_size: Optional[int] = None,  # Giới hạn theo ký tự (None = bỏ qua)
    chunk_overlap: int = 0,
    semantic_threshold: Optional[float] = None,
    min_sentences_per_chunk: int = 1,  # Số câu tối thiểu mỗi chunk (áp dụng cả hai chế độ)
    embedding_model: str = "all-MiniLM-L6-v2",
    device: Optional[object] = None,
    silent: bool = True,
    target_sentences_per_chunk: Optional[int] = None,  # Mục tiêu số câu mỗi chunk (ưu tiên hơn chunk_size)
    semantic_focus: bool = False,  # NEW: ignore target/count & chunk_size; cut only on semantic_threshold; no max size
    dynamic_semantic: bool = False,  # NEW: enable dynamic drop-based semantic segmentation
    dynamic_mode: str = "zscore",   # 'zscore' or 'relative'
    dynamic_k: float = 0.75,         # multiplier for std (zscore mode)
    dynamic_min_drop: float = 0.15,  # required drop previous->current (relative mode)
    dynamic_window: int = 5,         # moving average window (zscore adaptive baseline)
    max_sentences_per_chunk: Optional[int] = None,  # optional hard cap (force cut)
    dynamic_adaptive: bool = False,  # If True: fallback chain zscore -> relative -> percentile -> size based
    dynamic_percentile: float = 0.25, # Percentile threshold (0-1) for fallback low similarity cuts
    dynamic_log_cuts: bool = False,   # Log detailed cut reasoning
    fragment_long_sentences: bool = True, # NEW: further split very long sentences to increase boundary opportunities
    long_sentence_char_threshold: int = 400, # length above which we fragment
    approx_sentence_char_target: int = 320, # NEW: fallback char-based virtual sentence target when too few sentences
) -> Tuple[List[str], List[str], List[List[int]]]:
    """
    Optimized sentence-based text splitting with semantic control.
    
    Returns:
        - chunks: List of chunk texts
        - sentences: List of original sentences
        - sentence_groups: List of sentence indices for each chunk
    """
    
    # Extract sentences
    sentences = extract_sentences_spacy(text)
    # Optional fragmentation of very long sentences to increase semantic resolution
    if fragment_long_sentences and dynamic_semantic:
        refined: List[str] = []
        for s in sentences:
            if len(s) > long_sentence_char_threshold:
                # Try splitting on punctuation first, then coordinating conjunctions
                parts = re.split(r'(?<=[.;!?])\s+', s)
                # If still very long single part, split on comma groups
                split_parts: List[str] = []
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    if len(p) > long_sentence_char_threshold:
                        # secondary split on commas keeping semantic units (~75-120 chars)
                        accum = []
                        cur_len = 0
                        for seg in re.split(r',(\s+)', p):
                            if not seg or seg.isspace():
                                continue
                            if cur_len + len(seg) > 180 and accum:
                                split_parts.append(' '.join(accum).strip())
                                accum = [seg]
                                cur_len = len(seg)
                            else:
                                accum.append(seg)
                                cur_len += len(seg)
                        if accum:
                            split_parts.append(' '.join(accum).strip())
                    else:
                        split_parts.append(p)
                for sp in split_parts if split_parts else parts:
                    sp_clean = sp.strip()
                    if sp_clean:
                        if not re.search(r'[.!?]$', sp_clean):
                            sp_clean += '.'
                        refined.append(sp_clean)
            else:
                refined.append(s)
        if len(refined) > len(sentences):
            sentences = refined

    # Fallback coarse fragmentation: if still too few sentences for large doc, create virtual sentences
    if dynamic_semantic and len(sentences) < 8 and len(text) > approx_sentence_char_target * 4:
        virtual: List[str] = []
        for s in sentences:
            if len(s) <= approx_sentence_char_target * 1.3:
                virtual.append(s)
            else:
                for i in range(0, len(s), approx_sentence_char_target):
                    part = s[i:i+approx_sentence_char_target].strip()
                    if not part:
                        continue
                    if not re.search(r'[.!?]$', part):
                        part += '.'
                    virtual.append(part)
        if len(virtual) > len(sentences):
            if (dynamic_log_cuts or True) and not silent:
                print(f"[dynamic_info] coarse_fragment applied sentences {len(sentences)} -> {len(virtual)} (target~{approx_sentence_char_target} chars)")
            sentences = virtual

    # ---------------- New balanced sentence-count mode ----------------
    # Nếu đặt target_sentences_per_chunk thì ưu tiên chế độ chia theo số câu, đảm bảo không tạo chunk cuối quá nhỏ.
    if not semantic_focus and target_sentences_per_chunk is not None and target_sentences_per_chunk > 0:
        total_sent = len(sentences)
        if total_sent == 0:
            return [], [], []
        # Nếu tổng số câu < min_sentences_per_chunk => giữ nguyên 1 chunk
        if total_sent < min_sentences_per_chunk:
            return [text] if text.strip() else [], sentences, [list(range(total_sent))] if sentences else []

        # Tính số chunk ban đầu dựa trên target
        # k làm tròn gần nhất n / target
        k = max(1, round(total_sent / target_sentences_per_chunk))
        # Đảm bảo mỗi chunk >= min_sentences_per_chunk; nếu không thì giảm k
        while k > 1 and total_sent < k * min_sentences_per_chunk:
            k -= 1

        # Phân phối đều: các chunk đầu nhận (base_size+1) nếu còn dư
        base_size = total_sent // k
        extras = total_sent % k
        # Nếu base_size < min_sentences_per_chunk (trường hợp target < min) thì nâng lên
        if base_size < min_sentences_per_chunk:
            # Khi này k đã tối ưu nhỏ nhất đảm bảo mỗi chunk >= min_min, nên gom thành 1 chunk
            return [" ".join(sentences)], sentences, [list(range(total_sent))]

        chunk_sizes = []
        for i in range(k):
            size = base_size + (1 if i < extras else 0)
            chunk_sizes.append(size)

        # Sanity check cuối: nếu chunk cuối < min_sentences_per_chunk thì nhập vào chunk trước
        if chunk_sizes and chunk_sizes[-1] < min_sentences_per_chunk and len(chunk_sizes) > 1:
            chunk_sizes[-2] += chunk_sizes[-1]
            chunk_sizes.pop()

        # Tạo chunks
        chunks: List[str] = []
        sentence_groups: List[List[int]] = []
        cursor = 0
        for sz in chunk_sizes:
            group_indices = list(range(cursor, cursor + sz))
            chunk_text = " ".join(sentences[cursor:cursor + sz])
            chunks.append(chunk_text)
            sentence_groups.append(group_indices)
            cursor += sz

        return chunks, sentences, sentence_groups
    # ---------------- End new mode ----------------

    # Compute embeddings when semantic threshold or dynamic semantic is set
    embeddings: Optional[np.ndarray] = None
    if (semantic_threshold is not None or dynamic_semantic) and sentences:
        try:
            device_pref = _normalize_device(device)
            embeddings = embed_sentences_in_batches(sentences, model_name=embedding_model, base_batch_size=32, device=device_pref, silent=silent)
            if embeddings is not None and embeddings.size > 0:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                embeddings = embeddings / norms
            else:
                if not silent:
                    print("Semantic embedding returned empty; fallback to length-only splitter")
                semantic_threshold = None
                dynamic_semantic = False
                embeddings = None
        except Exception as e:
            if not silent:
                print(f"Semantic embedding failed ({e}); fallback to length-only splitter")
            semantic_threshold = None
            dynamic_semantic = False
            embeddings = None
    
    if not sentences:
        return [text] if text.strip() else [], [], [[0]] if text.strip() else []

    # Strict rule: if document has fewer sentences than minimum required per chunk,
    # skip chunking entirely (return no chunks) so that output contains only chunks meeting constraint.
    if len(sentences) < min_sentences_per_chunk:
        return [], sentences, []
    
    # Nếu không đặt giới hạn chunk_size, bỏ qua check “small enough”. (skip when dynamic mode or semantic_focus)
    if (not semantic_focus) and (not dynamic_semantic) and chunk_size is not None and len(text) <= chunk_size:
        sentence_indices = list(range(len(sentences)))
        return [text], sentences, [sentence_indices]

    # ========= Dynamic semantic drop-based segmentation =========
    if dynamic_semantic and embeddings is not None and len(sentences) > min_sentences_per_chunk:
        # Compute consecutive similarities s_i = sim(sent_{i-1}, sent_i)
        sims = [float(cosine_similarity(embeddings[i-1].reshape(1,-1), embeddings[i].reshape(1,-1))[0,0]) for i in range(1, len(sentences))]
        sims_arr = np.array(sims, dtype=float)
        mean_all = float(np.mean(sims_arr)) if sims_arr.size else 0.0
        std_all = float(np.std(sims_arr) + 1e-9)
        cut_indices: List[int] = []
        reasons: List[Tuple[int,str]] = []

        if (dynamic_log_cuts or True) and not silent:
            print(f"[dynamic_info] sentences={len(sentences)} sims={len(sims_arr)} mean={mean_all:.3f} std={std_all:.3f} min={(sims_arr.min() if sims_arr.size else 0):.3f} max={(sims_arr.max() if sims_arr.size else 0):.3f}")

        def apply_zscore(existing: List[int]):
            last_cut_local = existing[-1] if existing else 0
            for boundary in range(1, len(sentences)):
                if boundary - last_cut_local < min_sentences_per_chunk:
                    continue
                sim_prev = sims_arr[boundary-1]
                left = max(0, boundary-1-dynamic_window+1)
                window_vals = sims_arr[left:boundary]
                baseline_mean = float(np.mean(window_vals)) if window_vals.size else mean_all
                baseline_std = float(np.std(window_vals) + 1e-9)
                threshold_dynamic = baseline_mean - dynamic_k * baseline_std
                if sim_prev < threshold_dynamic:
                    existing.append(boundary)
                    reasons.append((boundary, f"zscore sim={sim_prev:.3f} < {threshold_dynamic:.3f} (mean={baseline_mean:.3f} std={baseline_std:.3f})"))
                    last_cut_local = boundary
                elif max_sentences_per_chunk and (boundary - last_cut_local) >= max_sentences_per_chunk:
                    existing.append(boundary)
                    reasons.append((boundary, f"hard_cap len={(boundary - last_cut_local)} >= {max_sentences_per_chunk}"))
                    last_cut_local = boundary

        def apply_relative(existing: List[int]):
            last_cut_local = existing[-1] if existing else 0
            for boundary in range(2, len(sentences)):
                if boundary - last_cut_local < min_sentences_per_chunk:
                    continue
                sim_prev = sims_arr[boundary-1]
                prev_sim = sims_arr[boundary-2]
                drop = prev_sim - sim_prev
                cut = False
                if drop >= dynamic_min_drop:
                    cut = True
                    reasons.append((boundary, f"relative drop={drop:.3f} >= {dynamic_min_drop}"))
                elif max_sentences_per_chunk and (boundary - last_cut_local) >= max_sentences_per_chunk:
                    cut = True
                    reasons.append((boundary, f"hard_cap len={(boundary - last_cut_local)} >= {max_sentences_per_chunk}"))
                if cut:
                    existing.append(boundary)
                    last_cut_local = boundary

        def apply_percentile(existing: List[int]):
            if sims_arr.size == 0:
                return
            last_cut_local = existing[-1] if existing else 0
            perc_thr = float(np.quantile(sims_arr, dynamic_percentile))
            for boundary in range(1, len(sentences)):
                if boundary - last_cut_local < min_sentences_per_chunk:
                    continue
                sim_prev = sims_arr[boundary-1]
                cut = False
                if sim_prev <= perc_thr:
                    cut = True
                    reasons.append((boundary, f"percentile sim={sim_prev:.3f} <= P{int(dynamic_percentile*100)}={perc_thr:.3f}"))
                elif max_sentences_per_chunk and (boundary - last_cut_local) >= max_sentences_per_chunk:
                    cut = True
                    reasons.append((boundary, f"hard_cap len={(boundary - last_cut_local)} >= {max_sentences_per_chunk}"))
                if cut:
                    existing.append(boundary)
                    last_cut_local = boundary

        if dynamic_mode == 'zscore':
            apply_zscore(cut_indices)
        elif dynamic_mode == 'relative':
            apply_relative(cut_indices)

        if dynamic_adaptive:
            if not cut_indices and dynamic_mode == 'zscore':
                apply_relative(cut_indices)
            if not cut_indices:
                apply_percentile(cut_indices)
            if not cut_indices and max_sentences_per_chunk:
                pos = max_sentences_per_chunk
                while pos < len(sentences):
                    cut_indices.append(pos)
                    reasons.append((pos, f"forced_cap_only len={max_sentences_per_chunk}"))
                    pos += max_sentences_per_chunk

        if (dynamic_log_cuts or True) and not silent and not cut_indices:
            print("[dynamic_info] No dynamic boundaries detected (stable coherence or too few sentences)")
            # Emergency: enforce window-based cuts if chunk would be extremely large
            if max_sentences_per_chunk and len(sentences) > max_sentences_per_chunk * 2:
                step = max_sentences_per_chunk
                synthetic = list(range(step, len(sentences), step))
                cut_indices.extend([c for c in synthetic if c < len(sentences)])
                reasons.append((-1, f"emergency_window_force every {step} sentences"))
            elif not max_sentences_per_chunk and len(sentences) >= 30:
                # If no hard cap specified but sentence count huge, force ~500 char based uniform splits
                # Estimate average sentence chars
                avg_len = sum(len(s) for s in sentences) / len(sentences)
                # target sentences per forced chunk ~ 6 sentences (adjust by average length)
                target = max(4, min(10, int(500 / max(avg_len, 1))))
                pos = target
                while pos < len(sentences):
                    cut_indices.append(pos)
                    reasons.append((pos, f"uniform_force size_target_sent={target}"))
                    pos += target

        cut_indices = sorted(set([c for c in cut_indices if 0 < c < len(sentences)]))

        if dynamic_log_cuts and not silent:
            for idx, reason in reasons:
                print(f"[dynamic_cut] boundary={idx} {reason}")
            if not reasons:
                print("[dynamic_cut] none (single-chunk fallback)")

        all_indices = cut_indices + [len(sentences)]
        chunks: List[str] = []
        sentence_groups: List[List[int]] = []
        cursor = 0
        for ci in all_indices:
            group = list(range(cursor, ci))
            if group:
                chunk_text = " ".join(sentences[cursor:ci])
                chunks.append(chunk_text)
                sentence_groups.append(group)
            cursor = ci
        # Merge trailing small chunk(s) to satisfy minimum sentence count constraint (last chunk)
        if sentence_groups and len(sentence_groups[-1]) < min_sentences_per_chunk and len(sentence_groups) > 1:
            while sentence_groups and len(sentence_groups[-1]) < min_sentences_per_chunk and len(sentence_groups) > 1:
                last = sentence_groups.pop()
                chunks.pop()
                sentence_groups[-1].extend(last)
                chunks[-1] = " ".join(sentences[sentence_groups[-1][0]: sentence_groups[-1][-1]+1])

        # Additional global pass: ensure NO chunk (not only last) violates min_sentences_per_chunk.
        # Strategy: single left-to-right pass merging any undersized chunk with its neighbor (prefer the shorter neighbor) until all satisfy.
        # This is defensive; dynamic boundary selection already tries to avoid creating these, but evaluation indicated rare <min cases (e.g., 3 sentences) due to
        # downstream sentence re-segmentation differences. This guarantees structural constraint at the source level.
        changed = True
        while changed and any(len(g) < min_sentences_per_chunk for g in sentence_groups if len(sentence_groups) > 1):
            changed = False
            i = 0
            while i < len(sentence_groups):
                if len(sentence_groups) == 1:
                    break
                if len(sentence_groups[i]) < min_sentences_per_chunk:
                    # Decide merge direction
                    if i == 0:
                        merge_with = 1
                    elif i == len(sentence_groups) - 1:
                        merge_with = i - 1
                    else:
                        # Prefer neighbor with fewer sentences to balance sizes
                        merge_with = i - 1 if len(sentence_groups[i-1]) <= len(sentence_groups[i+1]) else i + 1
                    # Normalize ordering (merge target into lower index)
                    a, b = (merge_with, i) if merge_with < i else (i, merge_with)
                    # Extend group a with b
                    sentence_groups[a].extend(sentence_groups[b])
                    # Rebuild chunk text
                    if b < len(chunks):
                        # Merge contiguous chunks: rebuild from sentence indices range
                        new_start = sentence_groups[a][0]
                        new_end = sentence_groups[a][-1]
                        chunks[a] = " ".join(sentences[new_start:new_end+1])
                        # Remove b
                        sentence_groups.pop(b)
                        chunks.pop(b)
                        changed = True
                        # Restart at previous index to catch cascades
                        i = max(0, a-1)
                        continue
                i += 1
        # If only one chunk remains and still below min (short document) we accept it.
        return chunks, sentences, sentence_groups
    # ========= End dynamic segmentation =========
    
    # Calculate sentence lengths
    sentence_lengths = [len(s) for s in sentences]
    
    chunks = []
    current_chunk_sentences = []
    current_size = 0
    sentence_groups = []
    current_group_indices = []
    
    # Nếu chunk_size None, đặt thành float('inf') để bỏ qua giới hạn độ dài
    effective_chunk_size = float('inf') if (chunk_size is None or semantic_focus) else chunk_size

    for i, sentence in enumerate(sentences):
        sentence_len = sentence_lengths[i]

        # (1) Semantic boundary check
        if (
            semantic_threshold is not None
            and current_chunk_sentences
            and embeddings is not None
            and len(current_chunk_sentences) >= min_sentences_per_chunk
        ):
            last_idx = current_group_indices[-1]
            sim_val = float(
                cosine_similarity(
                    embeddings[last_idx].reshape(1, -1),
                    embeddings[i].reshape(1, -1)
                )[0, 0]
            )
            if sim_val < semantic_threshold:
                # Commit current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(chunk_text)
                sentence_groups.append(list(current_group_indices))

                # Overlap handling
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

        # (2) Hard sentence too-long check (disabled in semantic_focus)
        if not semantic_focus and sentence_len > effective_chunk_size:
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(chunk_text)
                sentence_groups.append(list(current_group_indices))
                current_chunk_sentences = []
                current_group_indices = []
                current_size = 0
            chunks.append(sentence)
            sentence_groups.append([i])
            continue

        # (3) Size-based split (skipped if semantic_focus)
        projected_size = current_size + sentence_len + (1 if current_chunk_sentences else 0)
        if (
            (not semantic_focus)
            and projected_size > effective_chunk_size
            and current_chunk_sentences
            and len(current_chunk_sentences) >= min_sentences_per_chunk
        ):
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
            sentence_groups.append(list(current_group_indices))
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

        # (4) Append sentence
        current_chunk_sentences.append(sentence)
        current_group_indices.append(i)
        current_size = sum(len(s) for s in current_chunk_sentences)
        if len(current_chunk_sentences) > 1:
            current_size += len(current_chunk_sentences) - 1
    
    # Add final chunk if exists and meets minimum sentence requirement
    if current_chunk_sentences and len(current_chunk_sentences) >= min_sentences_per_chunk:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(chunk_text)
        sentence_groups.append(list(current_group_indices))
    elif current_chunk_sentences and len(current_chunk_sentences) < min_sentences_per_chunk:
        # Merge with previous chunk if it doesn't meet minimum
        if chunks and sentence_groups:
            # Merge with last chunk
            last_chunk_text = chunks[-1]
            last_sentence_group = sentence_groups[-1]
            
            merged_text = last_chunk_text + " " + " ".join(current_chunk_sentences)
            merged_indices = last_sentence_group + current_group_indices
            
            chunks[-1] = merged_text
            sentence_groups[-1] = merged_indices
        else:
            # No previous chunk to merge with, keep as is
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

## OIE formatting helper removed (OIE generation handled externally)

## Removed local process_oie_extraction & helper_save_raw_oie_data (use shared extract_oie_for_chunk & save_raw_oie_data)

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
    include_oie: bool = False,   # Kept for backward compatibility – ignored
    save_raw_oie: bool = False,  # Kept for backward compatibility – ignored
    output_dir: str = "./output",  # Unused now (no OIE saving) but retained for API stability
    device: Optional[object] = None,
    semantic_threshold: float = 0.30,
    min_sentences_per_chunk: int = 1,
    embedding_model: str = "all-MiniLM-L6-v2",
    silent: bool = False,
    target_sentences_per_chunk: Optional[int] = None,
    semantic_focus: bool = False,
    dynamic_semantic: bool = False,
    dynamic_mode: str = "zscore",
    dynamic_k: float = 0.75,
    dynamic_min_drop: float = 0.15,
    dynamic_window: int = 5,
    max_sentences_per_chunk: Optional[int] = None,
    dynamic_adaptive: bool = False,
    dynamic_percentile: float = 0.25,
    dynamic_log_cuts: bool = False,
    **kwargs 
) -> List[Tuple[str, str, Optional[str]]]:
    """Semantic text splitter.

    NOTE: OIE extraction has been removed from this method. The tuple third element
    (previously OIE string) is now always None so that downstream pipelines relying
    on the 3‑tuple structure remain compatible. Any request with include_oie=True
    will log a debug notice that OIE is externally generated.

    Returns:
        List[Tuple[str, str, Optional[str]]]: (chunk_id, chunk_text, None)
    """
    if include_oie and not silent:
        log_msg(silent, "OIE generation disabled in splitter (handled externally)", 'debug', 'split')

    log_msg(silent, f"Splitter doc={doc_id} size={chunk_size} overlap={chunk_overlap} min_sent={min_sentences_per_chunk} sem_th={semantic_threshold}", 'info', 'split')

    try:
        # Pre-count sentences to enforce hard minimum BEFORE any splitting so we can skip instead of fallback
        pre_sentences = extract_sentences_spacy(passage_text)
        if len(pre_sentences) < min_sentences_per_chunk:
            # New policy (user request): KEEP short documents as a SINGLE chunk (no splitting)
            if not silent:
                log_msg(silent, f"Doc short sentences={len(pre_sentences)} < min_sent={min_sentences_per_chunk} -> keep single chunk", 'info', 'split')
            chunk_hash = hashlib.sha1(passage_text.encode('utf-8', errors='ignore')).hexdigest()[:8]
            return [(f"{doc_id}_single_short_hash{chunk_hash}", passage_text, None)]

        # Split text into chunks using optimized algorithm
        log_msg(silent, "Splitting into chunks...", 'info', 'split')
        
        chunk_texts, sentences, sentence_groups = process_sentence_splitting_with_semantics(
            text=passage_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            semantic_threshold=semantic_threshold,
            min_sentences_per_chunk=min_sentences_per_chunk,
            embedding_model=embedding_model,
            device=device,
            silent=silent,
            target_sentences_per_chunk=target_sentences_per_chunk,
            semantic_focus=semantic_focus,
            dynamic_semantic=dynamic_semantic,
            dynamic_mode=dynamic_mode,
            dynamic_k=dynamic_k,
            dynamic_min_drop=dynamic_min_drop,
            dynamic_window=dynamic_window,
            max_sentences_per_chunk=max_sentences_per_chunk,
            dynamic_adaptive=dynamic_adaptive,
            dynamic_percentile=dynamic_percentile,
            dynamic_log_cuts=dynamic_log_cuts,
        )
        
        if not chunk_texts:
            # At this point we know the document had >= min sentences (otherwise we returned earlier).
            # An empty result means algorithm failed; we can safely fallback to full document only if it meets the minimum.
            if len(sentences) >= min_sentences_per_chunk:
                log_msg(silent, "No chunks generated – fallback to original text", 'warning', 'split')
                return [(f"{doc_id}_single_fallback", passage_text, None)]
            else:
                # Should not happen due to pre-check, but guard anyway.
                if not silent:
                    log_msg(silent, "Empty chunk list and sentence count below min – skipping", 'warning', 'split')
                return []
        
        log_msg(silent, f"Initial chunks={len(chunk_texts)}", 'info', 'split')

        # Build chunk tuples (OIE removed)
        chunks_with_oie: List[Tuple[str, str, Optional[str]]] = []

        for chunk_idx, chunk_text_content in enumerate(chunk_texts):
            oie_string: Optional[str] = None  # placeholder
            chunk_hash = hashlib.sha1(chunk_text_content.encode('utf-8', errors='ignore')).hexdigest()[:8]
            chunk_id = f"{doc_id}_chunk{chunk_idx}_hash{chunk_hash}"
            chunks_with_oie.append((chunk_id, chunk_text_content, oie_string))
        
        # Final validation and logging
        if not chunks_with_oie:
            log_msg(silent, "No valid chunks after processing – fallback", 'warning', 'split')
            return [(f"{doc_id}_fallback_complete", passage_text, None)]

        # Log final statistics
        token_counts = [_count_tokens_accurate(chunk[1]) for chunk in chunks_with_oie]
        log_msg(silent, f"Final chunks={len(chunks_with_oie)} tokens min={min(token_counts)} max={max(token_counts)} avg={sum(token_counts)/len(token_counts):.1f}", 'info', 'split')
        return chunks_with_oie

    except Exception as e:
        log_msg(silent, f"Splitter error doc={doc_id} {e}", 'error', 'split')
        return [(f"{doc_id}_error_fallback", passage_text, None)]

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
    semantic_threshold: float = 0.5,
    min_sentences_per_chunk: int = 1,
    embedding_model: str = "all-MiniLM-L6-v2",
    silent: bool = False,
    target_sentences_per_chunk: Optional[int] = None,
    semantic_focus: bool = False,
    dynamic_semantic: bool = False,
    dynamic_mode: str = "zscore",
    dynamic_k: float = 0.75,
    dynamic_min_drop: float = 0.15,
    dynamic_window: int = 5,
    max_sentences_per_chunk: Optional[int] = None,
    dynamic_adaptive: bool = False,
    dynamic_percentile: float = 0.25,
    dynamic_log_cuts: bool = False,
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
        semantic_threshold=semantic_threshold,
        min_sentences_per_chunk=min_sentences_per_chunk,
        embedding_model=embedding_model,
        silent=silent,
        target_sentences_per_chunk=target_sentences_per_chunk,
        semantic_focus=semantic_focus,
        dynamic_semantic=dynamic_semantic,
        dynamic_mode=dynamic_mode,
        dynamic_k=dynamic_k,
        dynamic_min_drop=dynamic_min_drop,
        dynamic_window=dynamic_window,
        max_sentences_per_chunk=max_sentences_per_chunk,
    dynamic_adaptive=dynamic_adaptive,
    dynamic_percentile=dynamic_percentile,
    dynamic_log_cuts=dynamic_log_cuts,
        **kwargs
    )
