from typing import List, Tuple, Optional
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
    *,
    min_sentences_per_chunk: int = 6,
    anchor_target_sentences: int = 6,
    anchor_similarity_floor: float = 0.0,
    embedding_batch_size: int = 64,
    device: Optional[str] = "cuda",
    silent: bool = False,
    **_extra,
) -> List[Tuple[str, str, Optional[str]]]:
    """Anchor greedy semantic grouping (non‑contiguous, MINIMAL PARAMETERS).

    Parameters:
        min_sentences_per_chunk: Minimum sentences per final chunk; short docs kept whole.
        anchor_target_sentences: Desired sentences per group (including anchor).
        anchor_similarity_floor: Optional minimum similarity to include a sentence under an anchor.
        embedding_batch_size: Embedding batch size passed to embedding helper.
        device: Device spec ("cuda" | "cpu" | "dml").
        silent: Suppress logging if True.
    Returns:
        List of (chunk_id, chunk_text, None)
    """
    if not silent:
        log_msg(False, f"[anchor_greedy] doc={doc_id} model={embedding_model}", 'info', 'group')

    sentences = extract_sentences_spacy(passage_text)
    if not sentences:
        return []

    if len(sentences) < min_sentences_per_chunk:
        chunk_hash = hashlib.sha1(passage_text.encode('utf-8', 'ignore')).hexdigest()[:8]
        return [(f"{doc_id}_short_hash{chunk_hash}", passage_text, None)]

    target_k = max(anchor_target_sentences, min_sentences_per_chunk)
    if len(sentences) <= target_k:
        chunk_hash = hashlib.sha1(passage_text.encode('utf-8', 'ignore')).hexdigest()[:8]
        return [(f"{doc_id}_single_hash{chunk_hash}", passage_text, None)]

    sim_matrix = create_similarity_matrix(
        sentences=sentences,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        device=normalize_device(device or 'cuda'),
        silent=silent,
    )
    if sim_matrix is None:
        chunk_hash = hashlib.sha1(passage_text.encode('utf-8', 'ignore')).hexdigest()[:8]
        return [(f"{doc_id}_matrix_fail_hash{chunk_hash}", passage_text, None)]

    n = len(sentences)
    centrality = ((sim_matrix.sum(axis=1) - 1.0) / (n - 1)).astype(float) if n > 1 else np.zeros(n, dtype=float)

    remaining = set(range(n))
    groups: List[List[int]] = []
    while remaining:
        anchor = max(remaining, key=lambda idx: centrality[idx])
        remaining.remove(anchor)
        # Rank remaining by similarity to anchor
        similars = [
            (idx, float(sim_matrix[anchor, idx]))
            for idx in remaining
            if float(sim_matrix[anchor, idx]) >= anchor_similarity_floor
        ]
        similars.sort(key=lambda x: x[1], reverse=True)
        need = max(0, target_k - 1)
        chosen = [idx for idx, _ in similars[:need]]
        for c in chosen:
            remaining.discard(c)
        groups.append(sorted([anchor] + chosen))

    # Merge trailing undersized group
    if len(groups) >= 2 and len(groups[-1]) < min_sentences_per_chunk:
        groups[-2].extend(groups[-1])
        groups.pop()

    # Forward attach any undersized groups
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
        return [(f"{doc_id}_fallback_hash{chunk_hash}", passage_text, None)]

    if not silent:
        log_msg(False, f"Anchor greedy chunks={len(final_chunks)} target={target_k}", 'info', 'group')
    return final_chunks


def semantic_chunk_passage_from_grouping_logic(
    doc_id: str,
    passage_text: str,
    embedding_model: str = "thenlper/gte-base",
    *,
    min_sentences_per_chunk: int = 6,
    anchor_target_sentences: int = 6,
    anchor_similarity_floor: float = 0.0,
    embedding_batch_size: int = 64,
    device: Optional[str] = "cuda",
    silent: bool = False,
    **_extra,
) -> List[Tuple[str, str, Optional[str]]]:
    return semantic_grouping_main(
        passage_text=passage_text,
        doc_id=doc_id,
        embedding_model=embedding_model,
        min_sentences_per_chunk=min_sentences_per_chunk,
        anchor_target_sentences=anchor_target_sentences,
        anchor_similarity_floor=anchor_similarity_floor,
        embedding_batch_size=embedding_batch_size,
        device=device,
        silent=silent,
    )
