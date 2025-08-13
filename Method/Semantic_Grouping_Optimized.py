from typing import List, Tuple, Optional, Union
import hashlib
import gc
import numpy as np

from Tool.Sentence_Segmenter import extract_sentences_spacy
from .semantic_common import (
    normalize_device,
    create_similarity_matrix,
    log_msg,
)

def semantic_grouping_main(
    passage_text: str,
    doc_id: str,
    embedding_model: str,
    initial_threshold: Union[str, float] = "auto",   # unused (compatibility)
    decay_factor: float = 0.85,                      # unused
    min_threshold: Union[str, float] = "auto",       # unused
    min_sentences_per_chunk: int = 1,
    include_oie: bool = False,                       # ignored
    save_raw_oie: bool = False,                      # ignored
    output_dir: str = "./output",                    # ignored
    initial_percentile: str = "85",                  # unused
    min_percentile: str = "25",                      # unused
    embedding_batch_size: int = 64,
    target_sentences_per_chunk: Optional[int] = None, # fallback for anchor target
    enforce_sentence_balancing: bool = False,         # unused
    sentence_target_tolerance: float = 0.3,           # unused
    semantic_focus: bool = True,                      # always semantic
    silent: bool = False,
    **kwargs
) -> List[Tuple[str, str, Optional[str]]]:
    """Anchor greedy semantic grouping (non-contiguous).

    Algorithm:
      1. Sentence segmentation.
      2. Similarity matrix via sentence embeddings.
      3. Centrality = mean similarity of each sentence to others.
      4. Repeatedly pick highest-centrality remaining sentence as anchor.
      5. Add top (k-1) most similar remaining sentences above optional similarity floor.
      6. Merge/attach undersized groups to satisfy min_sentences_per_chunk.
    """
    if not silent:
        log_msg(False, f"[anchor_greedy] doc={doc_id} model={embedding_model}", 'info', 'group')

    sentences = extract_sentences_spacy(passage_text)
    if not sentences:
        if not silent:
            log_msg(False, f"No sentences doc={doc_id}", 'warning', 'group')
        return []

    if len(sentences) < min_sentences_per_chunk:
        # Keep entire short document
        chunk_hash = hashlib.sha1(passage_text.encode('utf-8', 'ignore')).hexdigest()[:8]
        return [(f"{doc_id}_single_short_hash{chunk_hash}", passage_text, None)]

    target_k = (
        kwargs.get('anchor_target_sentences')
        or kwargs.get('target_sentences_per_chunk')
        or target_sentences_per_chunk
        or kwargs.get('target_sentences')
    )
    try:
        target_k = int(target_k) if target_k else None
    except Exception:
        target_k = None
    if not target_k or target_k <= 0:
        target_k = max(min_sentences_per_chunk, 8)

    if len(sentences) <= target_k:
        chunk_hash = hashlib.sha1(passage_text.encode('utf-8', 'ignore')).hexdigest()[:8]
        return [(f"{doc_id}_anchor_single_hash{chunk_hash}", passage_text, None)]

    sim_matrix = create_similarity_matrix(
        sentences=sentences,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=normalize_device(kwargs.get('device', 'cuda')),
        silent=silent,
    )
    if sim_matrix is None:
        if not silent:
            log_msg(False, "Similarity matrix failed -> single chunk", 'error', 'group')
        chunk_hash = hashlib.sha1(passage_text.encode('utf-8', 'ignore')).hexdigest()[:8]
        return [(f"{doc_id}_matrix_fail_hash{chunk_hash}", passage_text, None)]

    n = len(sentences)
    if n > 1:
        centrality = ((sim_matrix.sum(axis=1) - 1.0) / (n - 1)).astype(float)
    else:
        centrality = np.zeros(n, dtype=float)

    similarity_floor: float = float(kwargs.get('anchor_similarity_floor', 0.0) or 0.0)

    remaining = set(range(n))
    groups: List[List[int]] = []
    while remaining:
        anchor = max(remaining, key=lambda idx: centrality[idx])
        remaining.remove(anchor)
        candidates = []
        for idx in remaining:
            sim_val = float(sim_matrix[anchor, idx])
            if sim_val >= similarity_floor:
                candidates.append((idx, sim_val))
        candidates.sort(key=lambda x: x[1], reverse=True)
        take = max(0, target_k - 1)
        selected = [idx for idx, _ in candidates[:take]]
        for idx in selected:
            remaining.discard(idx)
        groups.append(sorted([anchor] + selected))

    # Merge trailing small group if exists
    if len(groups) >= 2 and len(groups[-1]) < min_sentences_per_chunk:
        groups[-2].extend(groups[-1])
        groups.pop()

    # Forward attach undersized groups
    merged: List[List[int]] = []
    buffer: List[int] = []
    for g in groups:
        if len(g) >= min_sentences_per_chunk:
            if buffer:
                g = sorted(buffer + g)
                buffer = []
            merged.append(g)
        else:
            buffer.extend(g)
    if buffer:
        if merged:
            merged[-1].extend(buffer)
        else:
            merged.append(buffer)

    if len(merged) >= 2 and len(merged[-1]) < min_sentences_per_chunk:
        merged[-2].extend(merged[-1])
        merged.pop()

    final_chunks: List[Tuple[str, str, Optional[str]]] = []
    for i, g in enumerate(merged):
        chunk_sentences = [sentences[idx] for idx in sorted(set(g)) if 0 <= idx < n]
        if not chunk_sentences:
            continue
        chunk_text = " ".join(chunk_sentences).strip()
        if not chunk_text:
            continue
        chunk_hash = hashlib.sha1(chunk_text.encode('utf-8', 'ignore')).hexdigest()[:8]
        final_chunks.append((f"{doc_id}_anchor{i}_hash{chunk_hash}", chunk_text, None))

    del sim_matrix
    gc.collect()

    if not final_chunks:
        chunk_hash = hashlib.sha1(passage_text.encode('utf-8', 'ignore')).hexdigest()[:8]
        return [(f"{doc_id}_anchor_fallback_hash{chunk_hash}", passage_text, None)]

    if not silent:
        log_msg(False, f"Anchor greedy chunks={len(final_chunks)} target_k={target_k}", 'info', 'group')
    return final_chunks


def semantic_chunk_passage_from_grouping_logic(
    doc_id: str,
    passage_text: str,
    embedding_model: str = "thenlper/gte-base",
    initial_threshold: Union[str, float] = "auto",
    decay_factor: float = 0.85,
    min_threshold: Union[str, float] = "auto",
    window_size: int = 3,  # unused
    embedding_batch_size: int = 64,
    include_oie: bool = False,
    save_raw_oie: bool = False,
    output_dir: str = "./output",
    initial_percentile: str = "85",
    min_percentile: str = "25",
    target_tokens: int = 120,  # unused
    tolerance: float = 0.25,   # unused
    min_sentences_per_chunk: int = 1,
    silent: bool = False,
    device: Optional[str] = "cuda",
    target_sentences_per_chunk: Optional[int] = None,
    enforce_sentence_balancing: bool = False,
    sentence_target_tolerance: float = 0.5,
    semantic_focus: bool = True,
    **kwargs
) -> List[Tuple[str, str, Optional[str]]]:
    # Force anchor mode semantics only
    kwargs['anchor_greedy_mode'] = True
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
        semantic_focus=semantic_focus,
        **kwargs,
    )
