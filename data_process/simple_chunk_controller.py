# -*- coding: utf-8 -*-
from __future__ import annotations

# Add parent directory to path for Method imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress noisy FutureWarnings & verbose library logs
import os, warnings, logging

# Disable tokenizers parallelism globally (must be before SentenceTransformer import)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Tăng giới hạn lên 1 triệu (hoặc lớn hơn nếu cần)

# For newer tokenizers versions explicitly disable parallelism API
try:
    import tokenizers
    tokenizers.set_parallelism(False)
except Exception:
    pass

# Ẩn cảnh báo encoder_attention_mask deprecated (transformers >=4.37)
warnings.filterwarnings(
    "ignore",
    message=r"`encoder_attention_mask` is deprecated.*BertSdpaSelfAttention.forward",
    category=FutureWarning,
)

# Giảm mức log của sentence_transformers / transformers để không in INFO
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import argparse
import csv
import sys
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict
from statistics import mean
try:
    import numpy as np
except Exception:
    np = None  # numpy optional for percentile stats

# Sentence & token utilities for evaluation
try:
    from Tool.Sentence_Segmenter import extract_sentences_spacy, count_tokens_spacy
except Exception:
    # Lightweight fallbacks
    import re
    def extract_sentences_spacy(text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        return [p.strip() for p in parts if p.strip()]
    def count_tokens_spacy(text: str) -> int:
        if not isinstance(text, str):
            return 0
        return len(re.findall(r'\b\w+\b', text))

# --- Standardized logging (reuse semantic_common utilities) ---
try:
    from Method.semantic_common import log_msg, init_logger
except Exception:
    # Fallback dummy if semantic_common not available
    def init_logger(name: str = "semantic.controller", level: int = logging.INFO):
        logger = logging.getLogger(name)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s', '%H:%M:%S'))
            logger.addHandler(h)
            logger.setLevel(level)
        return logger
    def log_msg(silent: bool, msg: str, level: str = 'info', component: str = None):
        if silent:
            return
        lvl_map = {'debug': logging.DEBUG,'info': logging.INFO,'warn': logging.WARNING,'warning': logging.WARNING,'error': logging.ERROR}
        lvl = lvl_map.get(level.lower(), logging.INFO)
        logging.getLogger(component or 'semantic.controller').log(lvl, msg)

CONTROLLER_SILENT: bool = False
def clog(msg: str, level: str = 'info'):
    """Controller log helper."""
    log_msg(CONTROLLER_SILENT, msg, level, 'controller')

def set_controller_log_level(level_name: str):
    level_map = {'debug': logging.DEBUG,'info': logging.INFO,'warn': logging.WARNING,'warning': logging.WARNING,'error': logging.ERROR}
    lvl = level_map.get(level_name.lower(), logging.INFO)
    init_logger('semantic.controller', lvl)

import pandas as pd
max_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_limit)
        break
    except OverflowError:
        max_limit = int(max_limit / 10)
# ==== Import optimized chunking methods ====
# NOTE: Inline OIE extraction removed from semantic methods. Third tuple element is always None.
# Run external OIE pipeline separately if needed.
# Semantic Splitter optimized – removed adaptive parameters (target_tokens, tolerance, enable_adaptive)
from Method.Semantic_Grouping_Optimized import (
    semantic_chunk_passage_from_grouping_logic as semantic_grouping,
)
from Method.Semantic_Splitter_Optimized import (
    chunk_passage_text_splitter as semantic_splitter,
)
from Method.Text_Splitter_Char_Naive import chunk_passage_text_splitter as chunk_passage_text_splitter_char_naive


"""
Usage Overview (MINIMAL MODE):

Active algorithms only:
    1. semantic_splitter_global  – Global adjacent sentence similarity valley detection.
             * Pure semantic boundaries (no character limits / overlaps)
             * Parameters (tune only if needed):
                     - min_sentences_per_chunk (default 6)
                     - global_z_k (default 0.50)
                     - global_min_drop (default 0.12)
                     - global_valley_prominence (default 0.04)
                     - target_sentences_per_chunk (optional approximate guidance)
                     - max_sentences_per_chunk (optional hard cap)

    2. semantic_grouping_anchor  – Non‑contiguous anchor greedy grouping.
             * Each chunk = anchor sentence (highest centrality not yet used) + top similar sentences
             * Parameters:
                     - min_sentences_per_chunk (default 6)
                     - anchor_target_sentences (default 6)
                     - anchor_similarity_floor (optional minimum similarity for inclusion)

Removed completely: dynamic splitter modes, z‑score streaming variants, size/character based splitting, overlap logic,
expansion / clustering grouping, all OIE logic. Third element in returned tuples is always None.

Examples:
    python simple_chunk_controller.py -i input.tsv -o output --configs semantic_splitter_global
    python simple_chunk_controller.py -i input.tsv -o output --configs semantic_grouping_anchor
    python simple_chunk_controller.py -i input.tsv -o output   (runs both by default)

Notes:
    - Sentence count is the ONLY control; short docs (< min_sentences_per_chunk) are preserved whole.
    - Use anchor grouping when you want semantic “bags” of related sentences (non‑contiguous).
    - Use splitter when you need contiguous segments respecting document order.
"""
# ==== Default constants ====
BATCH_SIZE = 600  # dòng/đợt đọc pandas
COMMON_DEFAULTS = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "device_preference": "dml",  # "cuda", "dml", hoặc "cpu"
    "enable_text_cleaning": True,  # Enable text cleaning by default
}

# ==== Text Cleaning for SpaCy Sentence Segmentation ========================

def preprocess_interview_format(text: str) -> str:
    """
    Tiền xử lý format interview/transcript để SpaCy có thể tách câu đúng.
    
    Args:
        text: Raw text potentially in interview format
        
    Returns:
        Preprocessed text with better sentence boundaries for SpaCy
    """
    import re
    
    if not isinstance(text, str):
        return ""
    
    # 1. Handle speaker attribution patterns: (Name) content -> Name said: "content"
    # This converts interview format to narrative format that SpaCy understands better
    
    # Pattern: (Speaker Name) followed by content with sentence ending
    # Example: (Gutierrez) The situation is complex. -> Gutierrez said: "The situation is complex."
    text = re.sub(r'\(([^)]+)\)\s+([A-Z][^.!?]*[.!?])', r'\1 said: "\2"', text)
    
    # 2. Handle speaker attribution without clear sentence ending
    # Pattern: (Name) content that continues... -> Name said: "content."
    text = re.sub(r'\(([^)]+)\)\s+([A-Z][^.!?]+?)(?=\s+\([^)]+\)|$)', r'\1 said: "\2."', text)
    
    # 3. Handle reporter attributions specifically
    # (Unidentified reporter) -> Reporter said:
    text = re.sub(r'\(Unidentified reporter\)\s+', 'Reporter said: "', text)
    text = re.sub(r'\(Reporter\)\s+', 'Reporter said: "', text)
    
    # 4. Handle "Here is a report by X:" patterns
    # Here is a report by Alvaro Ayala: (Ayala) content -> Here is a report by Alvaro Ayala: "content"
    text = re.sub(r'Here is a report by ([^:]+):\s+\([^)]+\)\s+', r'Here is a report by \1: "', text)
    
    # 5. CRITICAL: Remove empty speaker patterns that create meaningless chunks
    # Pattern: (Name). or (Name). (Name). etc.
    # This handles the problematic "(Unidentified reporter). (Reporter). (Reporter)." format
    text = re.sub(r'\([^)]+\)\.\s*', '', text)
    
    # 6. Close quotes that were opened but not closed
    # Count quotes and add closing quote if needed
    quote_count = text.count('"')
    if quote_count % 2 == 1:  # Odd number of quotes, need to close
        text += '"'
    
    # 7. Fix multiple spaces and clean up
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_document_for_spacy(text: str) -> str:
    """
    Làm sạch document để tránh SpaCy tách câu sai.
    Xử lý các vấn đề phổ biến: metadata, nested punctuation, quotes, dashes, lists.
    Tối ưu cho Robust04 dataset với patterns cụ thể.
    
    Args:
        text: Raw document text
        
    Returns:
        Cleaned text ready for SpaCy sentence segmentation
    """
    import re
    
    if not isinstance(text, str):
        return ""
    
    # === ROBUST04 SPECIFIC METADATA CLEANING ===
    
    # 1. Loại bỏ header metadata (Language: ... Article Type: ...) - fix cho format không có dấu cách
    # Pattern phải handle: "Language: Portuguese Article Type:BFN [Text] Sao Paulo --"
    text = re.sub(r'^Language:\s*\w+\s+Article Type:\s*[^\s\[\]]*\s*\[Text\]\s*', '', text, flags=re.IGNORECASE)
    
    # 2. Fallback pattern nếu không có [Text] tag
    text = re.sub(r'^Language:\s*\w+\s+Article Type:\s*[^\s]*\s*', '', text, flags=re.IGNORECASE)
    
    # 3. Loại bỏ các bracket metadata tags phổ biến - cẩn thận với whitespace
    # Loại bỏ metadata blocks
    text = re.sub(r'\[Article by[^\]]*\]\s*', '', text)
    text = re.sub(r'\[Report by[^\]]*\]\s*', '', text)
    text = re.sub(r'\[From the[^\]]*\]\s*', '', text)
    text = re.sub(r'\[Excerpts?\]\s*', '', text)
    text = re.sub(r'\[Text\]\s*', '', text)  # Tag cấu trúc - remove trailing whitespace
    text = re.sub(r'\[passage omitted\]\s*', '', text)
    text = re.sub(r'\[words indistinct\]\s*', '', text)
    
    # 4. Loại bỏ recording/transcript tags - cẩn thận với whitespace
    text = re.sub(r'\[Begin[^\]]*recording\]\s*', '', text)
    text = re.sub(r'\[end recording\]\s*', '', text)
    text = re.sub(r'\[Begin [^\]]*\]\s*', '', text)  # General begin tags
    text = re.sub(r'\[Interview with[^\]]*\]\s*', '', text)
    
    # 5. Xử lý reference brackets - convert thành contextual info
    text = re.sub(r'\[reference to[^\]]*\]\s*', '', text)
    
    # 6. Giữ lại nhưng normalize short references (< 30 chars) thành ()
    # Ví dụ: [parliament] -> (parliament), [Independence Day] -> (Independence Day)
    text = re.sub(r'\[([^\]]{1,30})\]', r'(\1)', text)
    
    # 7. Loại bỏ location/date metadata ở đầu nếu có pattern cụ thể
    # Ví dụ: "Santa Fe de Bogota, 28 Feb (DPA) --" có thể giữ vì là context
    # Chỉ loại nếu là pure metadata, không phải content
    
    # === GENERAL SENTENCE BOUNDARY FIXES ===
    
    # 8. Xử lý nested punctuation - Chuẩn hóa ngoặc lồng nhau
    text = re.sub(r'\(\s*([^()]*)\s*\[([^\]]*)\]\s*([^()]*)\)', r'(\1 \2 \3)', text)
    
    # 8. Xử lý abbreviations và capitals để tránh nhầm sentence boundary
    # CRITICAL: Fix acronym handling - common acronyms should not create sentence boundaries
    # First, protect common acronyms from being treated as sentence endings
    
    # Common political/military acronyms that appear with periods
    common_acronyms = [
        'ANC', 'SAP', 'APLA', 'SACP', 'MK', 'AWB', 'IFP', 'PAC', 'UDF',  # South African
        'FBI', 'CIA', 'DEA', 'ATF', 'NSA', 'DHS', 'DOJ', 'DOD',  # US agencies
        'NATO', 'UN', 'EU', 'OSCE', 'CSCE', 'CIS', 'CPRF', 'CPSU',  # International
        'PF', 'DPA', 'BFN', 'CSO', 'FBIS', 'ITAR', 'TASS',  # News/agencies
        'COCOM', 'DITA', 'QAP', 'KAM', 'SKAT', 'INPEC'  # Other organizations
    ]
    
    # Protect acronyms: Add temporary marker to prevent sentence boundary detection
    for acronym in common_acronyms:
        # "ANC. " -> "ANC__TEMP_DOT__ " (at end of sentence)
        text = re.sub(rf'\b{acronym}\.\s+([A-Z])', rf'{acronym}__TEMP_DOT__ \1', text)
        # "ANC." at end of text
        text = re.sub(rf'\b{acronym}\.$', rf'{acronym}__TEMP_DOT__', text)
        # "ANC. and" -> "ANC__TEMP_DOT__ and" (mid-sentence)
        text = re.sub(rf'\b{acronym}\.\s+([a-z])', rf'{acronym}__TEMP_DOT__ \1', text)
    
    # Handle general 2-4 letter acronyms more conservatively 
    # Only add periods for long acronyms (5+ letters) before lowercase
    text = re.sub(r'\b([A-Z]{5,})\b(?=\s+[a-z])', r'\1.', text)
    
    # 9. Chuẩn hóa dấu gạch ngang -- thành sentence boundary
    # NHƯNG KHÔNG áp dụng cho news dateline format "Location -- content"
    # Chỉ áp dụng khi có complete sentence trước --
    text = re.sub(r'([.!?])\s+--\s+([a-z])', r'\1 \2', text)  # After sentence ending
    text = re.sub(r'([.!?])\s+--\s+([A-Z])', r'\1 \2', text)  # After sentence ending
    
    # Xử lý -- ở giữa câu (không phải sentence boundary) - convert thành comma hoặc giữ nguyên
    # "mid-sentence content -- more content" -> "mid-sentence content, more content"  
    text = re.sub(r'([a-zA-Z])\s+--\s+([a-z])', r'\1, \2', text)
    # "Location -- Content" -> "Location: Content" cho news dateline format
    text = re.sub(r'([A-Z][a-zA-Z\s]+)\s+--\s+([A-Z])', r'\1: \2', text)
    
    # 10. Xử lý quotes phức tạp - normalize nested quotes
    text = re.sub(r'""([^"]*?)""', r'"\1"', text)
    text = re.sub(r'"([^"]*)"([^"]*)"([^"]*)"', r'"\1\2\3"', text)
    
    # 11. Xử lý numbered lists để tạo sentence boundaries
    # "conclusion: 1) Those who" -> "conclusion. Those who"
    text = re.sub(r':\s*(\d+\))\s*', r'. ', text)
    text = re.sub(r';\s*(\d+\))\s*', r'. ', text)
    
    # 12. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 13. Fix spacing around punctuation
    text = re.sub(r'\s+([.!?])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # 14. Remove empty lines và normalize paragraph breaks
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = ' '.join(lines)
    
    # 15. Fix sentence boundaries - conservative approach
    # Chỉ thêm dấu chấm khi có khoảng trống lớn
    text = re.sub(r'([a-z])\s{2,}([A-Z][a-z])', r'\1. \2', text)
    
    # 16. Clean up problematic period placements
    text = re.sub(r'([a-z])\.\s+([a-z])', r'\1 \2', text)
    text = re.sub(r'\bthe\.\s+([A-Z])', r'the \1', text)
    text = re.sub(r'\bin\.\s+([A-Z])', r'in \1', text)
    text = re.sub(r'\bof\.\s+([A-Z])', r'of \1', text)
    text = re.sub(r'\band\.\s+([A-Z])', r'and \1', text)
    
    # 17. Clean up multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
    # 18. RESTORE protected acronym periods
    # Convert temporary markers back to periods
    text = text.replace('__TEMP_DOT__', '.')
    
    # 19. Final cleanup
    text = text.strip()
    
    return text


def validate_cleaned_text(original: str, cleaned: str, max_length_diff: float = 0.3) -> bool:
    """
    Kiểm tra xem text cleaning có quá aggressive không.
    
    Args:
        original: Original text
        cleaned: Cleaned text  
        max_length_diff: Maximum allowed length difference ratio
        
    Returns:
        True if cleaning is reasonable, False if too aggressive
    """
    if not original or not cleaned:
        return False
    
    # Check length difference
    length_ratio = abs(len(cleaned) - len(original)) / len(original)
    if length_ratio > max_length_diff:
        print(f"Warning: Text cleaning removed {length_ratio:.1%} of content")
        return False
    
    # Check if essential content remains
    words_original = len(original.split())
    words_cleaned = len(cleaned.split())
    word_ratio = abs(words_cleaned - words_original) / words_original
    
    if word_ratio > max_length_diff:
        print(f"Warning: Text cleaning removed {word_ratio:.1%} of words")
        return False
    
    return True

# ==== Worker initializer =====================================================

def _worker_init(model_name: str, device_pref: str):
    """Nạp SentenceTransformer + spaCy 1 lần cho mỗi process."""
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers import SentenceTransformer
    import spacy

    global _GLOBALS  # type: ignore
    _GLOBALS = {}

    # spaCy sentencizer tối giản với custom preprocessing
    _GLOBALS["nlp"] = spacy.blank("en").add_pipe("sentencizer")

    # xác định device
    def _resolve(pref: str):
        import torch
        pref = (pref or "").lower()
        if pref in {"dml", "directml"}:
            try:
                import torch_directml

                return torch_directml.device() if torch_directml.is_available() else "cpu"
            except Exception:
                return "cpu"
        if pref == "cuda":
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                return torch.device("cuda")
            return torch.device(pref)
        if pref in {"", "cpu"}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return pref

    device_obj = _resolve(device_pref)

    clog(f"Worker init device={device_obj}", 'debug')
    _GLOBALS["model"] = SentenceTransformer(model_name, device=device_obj)


# ==== Helper: chunk 1 batch ==================================================

def _process_batch(
    batch_df: pd.DataFrame,
    config: Dict[str, Any],
    batch_idx: int,
    embedding_model: str,
    device_preference: str,
    enable_text_cleaning: bool = True,
    max_chunk_chars: Optional[int] = 50000,
    collect_metadata: bool = False,
) -> List[List[Any]]:
    """Chunk các dòng trong batch và trả về list row cho TSV output."""
    results: List[List[Any]] = []

    # Handle both multiprocessing and single-threaded execution
    try:
        global _GLOBALS  # type: ignore
        nlp = _GLOBALS.get("nlp")
        model = _GLOBALS.get("model")
        if model is None:
            raise RuntimeError("_worker_init chưa chạy – model None")
    except (NameError, AttributeError):
        # Single-threaded execution - initialize locally
        import spacy
        from sentence_transformers import SentenceTransformer
        import torch
        
        nlp = spacy.blank("en").add_pipe("sentencizer")
        
        # Resolve device for single-threaded execution
        if device_preference.lower() in {"dml", "directml"}:
            try:
                import torch_directml
                device_obj = torch_directml.device() if torch_directml.is_available() else "cpu"
            except Exception:
                device_obj = "cpu"
        elif device_preference.lower() == "cuda":
            device_obj = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_obj = device_preference.lower()
        
        print(f"[Single-threaded] Using device: {device_obj}")
        model = SentenceTransformer(embedding_model, device=device_obj)

    method_choice = config["method_choice"]
    params = dict(config["params"])  # shallow copy

    # map
    func_map = {"1": semantic_grouping, "2": semantic_splitter, "3": chunk_passage_text_splitter_char_naive}
    chunk_func = func_map[method_choice]

    import os

    pid = os.getpid()
    clog(f"PID {pid} batch={batch_idx} rows={len(batch_df)} start", 'debug')

    for local_idx, row in batch_df.iterrows():
        # Input header: query_id | query_text | document_id | document | label
        try:
            query_id = row["query_id"]
            passage  = row["document"]
            doc_orig = row["document_id"]
            label    = row["label"]
        except KeyError:
            # fallback positional (cũ) để không vỡ khi file thiếu header mới
            query_id, _qtxt, doc_orig, passage, label = row.iloc[0:5]

        # === TEXT CLEANING FOR SPACY SENTENCE SEGMENTATION ===
        # Apply cleaning to improve SpaCy sentence detection
        if enable_text_cleaning:
            original_passage = passage
            
            # Step 1: Clean document metadata and formatting
            cleaned_passage = clean_document_for_spacy(passage)
            
            # Step 2: Preprocess interview/transcript format for better sentence segmentation
            interview_processed = preprocess_interview_format(cleaned_passage)
            
            # Validate cleaning didn't remove too much content
            if not validate_cleaned_text(original_passage, interview_processed):
                clog(f"Text cleaning aggressive doc={doc_orig} reverting original", 'warning')
                interview_processed = original_passage
            
            # Use preprocessed text for processing
            passage = interview_processed

        if not isinstance(passage, str) or not passage.strip():
            continue

        # TREC ROBUST04 FILTER: Skip documents marked as having no information
        if passage.strip() == "This document has no information.":
            clog(f"Skip empty-info doc={doc_orig}", 'debug')
            continue

        # doc_id duy nhất cho thuật toán chunking (không dùng làm output)
        doc_id = f"{doc_orig}_B{batch_idx}_{local_idx}"

        if method_choice in {"1", "2"}:  # semantic methods yêu cầu embedding
            extra = {
                "embedding_model": embedding_model,
                "device": device_preference,
            }
        else:
            extra = {}

        # All semantic methods run in silent mode unless controller log level adjusted
        local_silent = True

        tuples: List[Tuple[str, str, Optional[str]]] = chunk_func(
            doc_id=doc_id,
            passage_text=passage,
            **params,
            **extra,
            silent=local_silent,
            output_dir=None,
            collect_metadata=collect_metadata,
        )
        
        # Enhanced logging for chunk processing results
        chunk_sizes = [len(chunk_text) for _, chunk_text, _ in tuples if chunk_text]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            max_size = max(chunk_sizes)
            min_size = min(chunk_sizes)
            
            # Log statistics every 100 documents for monitoring
            # Adaptive logging frequency: for small batches (<200 docs) log every doc; otherwise every 100
            log_every = 1 if len(batch_df) <= 200 else 100
            if local_idx % log_every == 0:
                method_name = {
                    "1": "semantic_grouping",
                    "2": "semantic_splitter"
                }.get(method_choice, "unknown")
                
                balanced_info = ""
                # Removed char-size control: only sentence-based statistics remain
                
                clog(f"doc_idx={local_idx} method={method_name} chunks={len(tuples)} size_avg={avg_size:.0f} min={min_size} max={max_size}{balanced_info}", 'info')
        
        # Append chunks for THIS document
        for chunk_id, chunk_text, meta in tuples:
            if not isinstance(chunk_text, str):
                chunk_text = str(chunk_text)
            chunk_text = (chunk_text.replace('\t', ' ')  # tabs -> space
                                      .replace('\n', ' ')  # newlines -> space
                                      .replace('\r', ' ')  # CR -> space
                                      .strip())
            if not chunk_text:
                continue
            if max_chunk_chars is not None and max_chunk_chars > 0 and len(chunk_text) > max_chunk_chars:
                clog(f"truncate chunk chars={len(chunk_text)} -> {max_chunk_chars}", 'warning')
                chunk_text = chunk_text[:max_chunk_chars] + "..."
            if collect_metadata and meta:
                results.append([query_id, doc_orig, chunk_text, label, meta])
            else:
                results.append([query_id, doc_orig, chunk_text, label])
    clog(f"batch={batch_idx} done chunks={len(results)}", 'debug')
    return results


# ==== Controller per configuration ==========================================

def run_config(
    config: Dict[str, Any],
    input_path: Path,
    output_dir: Path,
    config_workers: int = 1,
    batch_size: int = BATCH_SIZE,
    embedding_model: Optional[str] = None,
    device_preference: Optional[str] = None,
    enable_text_cleaning: Optional[bool] = None,
    max_chunk_chars: Optional[int] = 50000,
    collect_metadata: bool = False,
):
    """Chạy chunking cho 1 cấu hình."""
    import time

    # Use provided parameters or fall back to defaults - USER PARAMETERS TAKE PRIORITY
    actual_embedding_model = embedding_model if embedding_model is not None else COMMON_DEFAULTS["embedding_model"]
    actual_device_preference = device_preference if device_preference is not None else COMMON_DEFAULTS["device_preference"]
    actual_enable_text_cleaning = enable_text_cleaning if enable_text_cleaning is not None else COMMON_DEFAULTS["enable_text_cleaning"]

    start = time.time()
    outfile = output_dir / f"{config['name']}_chunks.tsv"
    map_file = output_dir / f"{config['name']}_chunk_map.tsv" if collect_metadata else None

    # đọc input theo chunk
    with pd.read_csv(
        input_path,
        sep="\t",
        header=0,
        chunksize=batch_size,
        engine="python",
        on_bad_lines="warn",
        quoting=csv.QUOTE_NONE,
        dtype=str,
        encoding_errors="replace",
    ) as reader:
        all_batches = [(idx + 1, df) for idx, df in enumerate(reader)]

    # Initialize output file with header
    outfile.parent.mkdir(exist_ok=True)
    total_chunks = 0
    
    eval_file = output_dir / f"{config['name']}_eval_chunks.tsv"
    summary_file = output_dir / f"{config['name']}_eval_summary.txt"

    # Accumulators for evaluation stats
    eval_sent_counts: List[int] = []
    eval_word_counts: List[int] = []
    eval_token_counts: List[int] = []
    eval_char_counts: List[int] = []
    doc_chunk_counter: Dict[str, int] = {}

    def _percentiles(values: List[int], ps=(50, 90, 95)) -> Dict[str, float]:
        if not values:
            return {f"p{p}": 0 for p in ps}
        if np is None:
            # simple manual approach
            sorted_vals = sorted(values)
            out = {}
            for p in ps:
                if not sorted_vals:
                    out[f"p{p}"] = 0
                else:
                    k = int(round((p/100)*(len(sorted_vals)-1)))
                    out[f"p{p}"] = float(sorted_vals[k])
            return out
        arr = np.array(values)
        return {f"p{p}": float(np.percentile(arr, p)) for p in ps}

    with open(outfile, "w", encoding="utf-8", newline="") as f, open(eval_file, "w", encoding="utf-8", newline="") as feval, (open(map_file, "w", encoding="utf-8", newline="") if map_file else open(os.devnull, 'w')) as fmap:
        w = csv.writer(f, delimiter="\t")
        main_header = ["query_id", "document_id", "chunk_text", "label"]
        if collect_metadata:
            main_header.append("meta_json")
        w.writerow(main_header)
        w_eval = csv.writer(feval, delimiter="\t")
        w_eval.writerow(["query_id", "document_id", "sentences", "words", "tokens", "chars", "label"])  # per-chunk metrics
        w_map = csv.writer(fmap, delimiter='\t') if collect_metadata and w_map else None
        if collect_metadata and w_map:
            w_map.writerow(["query_id", "document_id", "chunk_id", "sent_indices", "n_sent", "sim_mean", "sim_min", "sim_max", "sim_std", "anchor", "anchor_centrality"])  # some fields may be blank
        
        # pool xử lý batch with streaming write
        if config_workers > 1:
            with ProcessPoolExecutor(
                max_workers=config_workers,
                initializer=_worker_init,
                initargs=(actual_embedding_model, actual_device_preference),
            ) as exe:
                futures = {
                    exe.submit(_process_batch, df, config, idx, actual_embedding_model, actual_device_preference, actual_enable_text_cleaning, max_chunk_chars, collect_metadata): idx 
                    for idx, df in all_batches
                }
                
                # Write results as they complete (streaming)
                completed_batches = 0
                for fut in as_completed(futures):
                    try:
                        batch_results = fut.result()
                        if batch_results:
                            # CRITICAL: Validate each row before writing to prevent TSV corruption
                            validated_results = []
                            for row in batch_results:
                                try:
                                    # Ensure all fields are strings and properly formatted
                                    validated_row = []
                                    for i in range(len(row)):
                                        validated_row.append(str(row[i]) if row[i] is not None else "")
                                    validated_results.append(validated_row)
                                except Exception as row_error:
                                    print(f"[WARNING] Skipping invalid row: {row_error}")
                                    continue
                            
                            if validated_results:
                                # Write main chunks
                                w.writerows(validated_results)
                                # Evaluation per chunk
                                for row in validated_results:
                                    qid, doc_id_row, chunk_text_row, lbl = row[0:4]
                                    # Update doc chunk counter
                                    doc_chunk_counter[doc_id_row] = doc_chunk_counter.get(doc_id_row, 0) + 1
                                    sent_ct = len(extract_sentences_spacy(chunk_text_row))
                                    word_ct = len(chunk_text_row.split())
                                    tok_ct = count_tokens_spacy(chunk_text_row)
                                    char_ct = len(chunk_text_row)
                                    eval_sent_counts.append(sent_ct)
                                    eval_word_counts.append(word_ct)
                                    eval_token_counts.append(tok_ct)
                                    eval_char_counts.append(char_ct)
                                    w_eval.writerow([qid, doc_id_row, sent_ct, word_ct, tok_ct, char_ct, lbl])
                                    if collect_metadata and w_map and len(row) >=5 and row[4]:
                                        # parse metadata JSON
                                        try:
                                            import json
                                            meta = json.loads(row[4])
                                        except Exception:
                                            meta = {}
                                        w_map.writerow([
                                            qid,
                                            doc_id_row,
                                            meta.get("chunk_id", ""),
                                            meta.get("sent_indices", ""),
                                            meta.get("n", ""),
                                            meta.get("sim_mean", ""),
                                            meta.get("sim_min", ""),
                                            meta.get("sim_max", ""),
                                            meta.get("sim_std", ""),
                                            meta.get("anchor", ""),
                                            meta.get("anchor_centrality", ""),
                                        ])
                                f.flush(); feval.flush()
                                if collect_metadata and fmap:
                                    fmap.flush()
                                total_chunks += len(validated_results)
                        completed_batches += 1
                        
                        # Progress update
                        if completed_batches % max(1, len(all_batches) // 10) == 0:
                            progress = (completed_batches / len(all_batches)) * 100
                            clog(f"config={config['name']} progress={progress:.1f}% batches={completed_batches}/{len(all_batches)} chunks={total_chunks}", 'info')
                    except Exception as e:
                        clog(f"batch error type={type(e).__name__} msg={e}", 'error')
                        completed_batches += 1
        else:
            # Single-threaded with streaming write
            for batch_num, (idx, df) in enumerate(all_batches, 1):
                try:
                    batch_results = _process_batch(df, config, idx, actual_embedding_model, actual_device_preference, actual_enable_text_cleaning, max_chunk_chars, collect_metadata)
                    if batch_results:
                        # CRITICAL: Validate each row before writing to prevent TSV corruption
                        validated_results = []
                        for row in batch_results:
                            try:
                                validated_row = [str(col) if col is not None else "" for col in row]
                                validated_results.append(validated_row)
                            except Exception as row_error:
                                print(f"[WARNING] Skipping invalid row: {row_error}")
                                continue
                        
                        if validated_results:
                            w.writerows(validated_results)
                            for row in validated_results:
                                qid, doc_id_row, chunk_text_row, lbl = row[0:4]
                                doc_chunk_counter[doc_id_row] = doc_chunk_counter.get(doc_id_row, 0) + 1
                                sent_ct = len(extract_sentences_spacy(chunk_text_row))
                                word_ct = len(chunk_text_row.split())
                                tok_ct = count_tokens_spacy(chunk_text_row)
                                char_ct = len(chunk_text_row)
                                eval_sent_counts.append(sent_ct)
                                eval_word_counts.append(word_ct)
                                eval_token_counts.append(tok_ct)
                                eval_char_counts.append(char_ct)
                                w_eval.writerow([qid, doc_id_row, sent_ct, word_ct, tok_ct, char_ct, lbl])
                                if collect_metadata and w_map and len(row) >=5 and row[4]:
                                    try:
                                        import json
                                        meta = json.loads(row[4])
                                    except Exception:
                                        meta = {}
                                    w_map.writerow([
                                        qid,
                                        doc_id_row,
                                        meta.get("chunk_id", ""),
                                        meta.get("sent_indices", ""),
                                        meta.get("n", ""),
                                        meta.get("sim_mean", ""),
                                        meta.get("sim_min", ""),
                                        meta.get("sim_max", ""),
                                        meta.get("sim_std", ""),
                                        meta.get("anchor", ""),
                                        meta.get("anchor_centrality", ""),
                                    ])
                            f.flush(); feval.flush()
                            if collect_metadata and fmap:
                                fmap.flush()
                            total_chunks += len(validated_results)
                        
                        # Progress update
                        if batch_num % max(1, len(all_batches) // 10) == 0:
                            progress = (batch_num / len(all_batches)) * 100
                            clog(f"config={config['name']} progress={progress:.1f}% batches={batch_num}/{len(all_batches)} chunks={total_chunks}", 'info')
                except Exception as e:
                    clog(f"batch={batch_num} error type={type(e).__name__} msg={e}", 'error')
                    continue

    elapsed = time.time() - start
    
    # Enhanced completion message with statistics
    # Correct average: divide by actual number of unique processed documents (doc_chunk_counter)
    avg_chunks_per_doc = total_chunks / max(1, len(doc_chunk_counter))
    chunks_per_sec = total_chunks / max(1, elapsed)
    
    split_info = ""  # character-based split info removed
    
    preserve_info = ""
    config_params = config.get('params', {})
    if config_params.get('min_sentences_per_chunk', 2) == 1:
        preserve_info = " | CONTENT_PRESERVED"
    
    # ----- Evaluation summary -----
    unique_docs = len(doc_chunk_counter)
    chunks_per_doc = list(doc_chunk_counter.values())
    def _summ(name: str, values: List[int]) -> str:
        if not values:
            return f"{name}: count=0"
        pct = _percentiles(values)
        return (f"{name}: count={len(values)} min={min(values)} max={max(values)} "
                f"mean={mean(values):.2f} median={pct['p50']:.2f} p90={pct['p90']:.2f} p95={pct['p95']:.2f}")
    summary_lines = [
        f"CONFIG: {config['name']}",
        f"Total chunks: {total_chunks}",
        f"Unique documents: {unique_docs}",
        _summ("Sentences per chunk", eval_sent_counts),
        _summ("Words per chunk", eval_word_counts),
        _summ("Tokens per chunk", eval_token_counts),
        _summ("Chars per chunk", eval_char_counts),
        _summ("Chunks per document", chunks_per_doc),
    ]
    try:
        with open(summary_file, 'w', encoding='utf-8') as sf:
            sf.write("\n".join(summary_lines))
    except Exception as e:
        clog(f"Failed to write evaluation summary: {e}", 'error')

    extra_map = f" map={map_file}" if collect_metadata else ""
    clog(f"config={config['name']} done file={outfile} eval={eval_file} summary={summary_file}{extra_map}", 'info')
    clog(f"stats chunks={total_chunks} elapsed={elapsed:.1f}s rate={chunks_per_sec:.1f}/s avg_per_doc={avg_chunks_per_doc:.2f}{split_info}{preserve_info}", 'info')


# ==== RUN_CONFIGURATIONS =====================================================


RUN_CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "name": "semantic_splitter_global",
        "method_choice": "2",
        "params": {
            "min_sentences_per_chunk": 6,
            "global_z_k": 0.40,
            "global_min_drop": 0.08,
            "global_valley_prominence": 0.04,
            "target_sentences_per_chunk": 6,
            "max_sentences_per_chunk": 18,
            "global_debug": False,
        },
        "description": "Contiguous semantic splitter using global adjacent similarity valleys (chunks ≥6 sentences)"
    },
    {
        "name": "semantic_grouping_anchor",
        "method_choice": "1",
        "params": {
            "min_sentences_per_chunk": 6,
            "anchor_target_sentences": 6,
            # Optional: "anchor_similarity_floor": 0.0,
        },
        "description": "Non‑contiguous anchor greedy grouping (hub sentence + top similar) (chunks ≥6 sentences)"
    },
    {
        "name": "text_splitter_char_naive",
        "method_choice": "3",
        "params": {
            "chunk_size": 600,
            "overlap": 0,
        },
        "description": "Naive fixed-character splitter (no semantics)"
    },
]

_MODEL_PRESETS = {
    "1": ("thenlper/gte-large", "High Quality (yêu cầu ≥4.5GB VRAM)"),
    "2": ("thenlper/gte-base", "Balanced (≈2GB VRAM)"),
    "3": ("sentence-transformers/all-MiniLM-L12-v2", "Fast (≈0.7GB VRAM)"),
    "4": ("sentence-transformers/all-MiniLM-L6-v2", "Ultra-Fast (≈0.5GB VRAM)"),
    "5": ("auto", "Auto-select based on sys RAM/GPU"),
}


def _recommend_auto_model() -> str:
    try:
        import psutil

        avail = psutil.virtual_memory().available / (1024 ** 3)
        if avail >= 12:
            return "thenlper/gte-base"
        elif avail >= 6:
            return "sentence-transformers/all-MiniLM-L12-v2"
        else:
            return "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        return "sentence-transformers/all-MiniLM-L6-v2"


def interactive_ui():
    input_path = Path(input("Enter path to input TSV (query_id\tquery_text\tdocument_id\tdocument\tlabel): ").strip())
    if not input_path.exists():
        clog("interactive: input file not found", 'error')
        return

    out_dir = Path(input("Enter output directory [training_datasets]: ").strip() or "training_datasets")
    out_dir.mkdir(exist_ok=True)

    # ----- Model preset -----
    current_embedding_model = COMMON_DEFAULTS["embedding_model"]
    print(f"\nCurrent embedding model: {current_embedding_model}")
    if input("Change model? (y/N): ").strip().lower() == "y":
        print("\nQUICK MODEL PRESETS:")
        for key, (model_name, desc) in _MODEL_PRESETS.items():
            print(f"  {key}. {model_name} - {desc}")  # keep user prompt prints
        choice = input("Select preset (1-5) or press ENTER to keep current: ").strip()
        if choice in _MODEL_PRESETS:
            selected = _MODEL_PRESETS[choice][0]
            if selected == "auto":
                selected = _recommend_auto_model()
            current_embedding_model = selected
            clog(f"interactive: embedding_model={selected}", 'info')

    # ----- Config selection -----
    print("\nAvailable configurations:")
    for i, cfg in enumerate(RUN_CONFIGURATIONS, 1):
        desc = cfg.get('description', 'No description')
        params_info = []
        params = cfg.get('params', {})
        if 'min_sentences_per_chunk' in params:
            params_info.append(f"min_sentences={params['min_sentences_per_chunk']}")
        if 'target_sentences_per_chunk' in params:
            params_info.append(f"target_sent={params['target_sentences_per_chunk']}")
        if 'anchor_target_sentences' in params:
            params_info.append(f"anchor_target={params['anchor_target_sentences']}")
        param_str = f" ({', '.join(params_info)})" if params_info else ""
        print(f"  {i:2d}. {cfg['name']}{param_str}")
        print(f"      {desc}")
    
    cfg_input = input("\nEnter config numbers (comma-separated) or names or ENTER for ALL: ").strip()
    
    if cfg_input:
        # Handle both numbers and names
        selected_cfgs = []
        for item in cfg_input.split(','):
            item = item.strip()
            if item.isdigit() and 1 <= int(item) <= len(RUN_CONFIGURATIONS):
                selected_cfgs.append(RUN_CONFIGURATIONS[int(item) - 1])
            else:
                # Try to match by name
                matching = [c for c in RUN_CONFIGURATIONS if c["name"] == item]
                selected_cfgs.extend(matching)
    else:
        selected_cfgs = RUN_CONFIGURATIONS
    
    if not selected_cfgs:
        clog("interactive: no configuration selected", 'warning')
        return

    # ----- Workers per config (batch) -----
    cfg_workers = int(input("Workers per config (batch) [1]: ") or "1")
    max_chunk_chars_input = input("Max chunk chars before truncation [50000, 0=disable]: ").strip()
    max_chunk_chars = 50000 if max_chunk_chars_input == "" else int(max_chunk_chars_input)

    print("\n=== SUMMARY ===")
    print(f"Input        : {input_path}")
    print(f"Output dir   : {out_dir}")
    print(f"Embedding    : {current_embedding_model}")
    print(f"Configurations: {len(selected_cfgs)} selected")
    for cfg in selected_cfgs:
        params_summary = []
        params = cfg.get('params', {})
        if 'min_sentences_per_chunk' in params:
            params_summary.append(f"min_sentences={params['min_sentences_per_chunk']}")
        if 'target_sentences_per_chunk' in params:
            params_summary.append(f"target_sent={params['target_sentences_per_chunk']}")
        if 'anchor_target_sentences' in params:
            params_summary.append(f"anchor_target={params['anchor_target_sentences']}")
        param_str = f" ({', '.join(params_summary)})" if params_summary else ""
        print(f"  - {cfg['name']}{param_str}")
    print(f"Workers      : per-config={cfg_workers} max_chunk_chars={max_chunk_chars}\n")

    # spawn start-method
    mp.set_start_method("spawn", force=True)

    # Chạy tất cả configs được chọn - use user selections, not COMMON_DEFAULTS
    for i, cfg in enumerate(selected_cfgs, 1):
        clog(f"interactive run config {i}/{len(selected_cfgs)} name={cfg['name']}", 'info')
        run_config(
            config=cfg,
            input_path=input_path,
            output_dir=out_dir,
            config_workers=cfg_workers,
            batch_size=BATCH_SIZE,
            embedding_model=current_embedding_model,
            device_preference=COMMON_DEFAULTS["device_preference"],
            enable_text_cleaning=COMMON_DEFAULTS["enable_text_cleaning"],
            max_chunk_chars=max_chunk_chars,
        )

    clog(f"interactive all configs completed count={len(selected_cfgs)} out={out_dir}", 'info')


# ==== CLI ====================================================================

def main():
    # Display available configurations in help with enhanced descriptions
    config_list = []
    for i, cfg in enumerate(RUN_CONFIGURATIONS):
        name = cfg['name']
        desc = cfg.get('description', 'No description')
        params = cfg.get('params', {})
        features = []
        if 'target_sentences_per_chunk' in params:
            features.append(f"TARGET≈{params['target_sentences_per_chunk']}")
        if 'anchor_target_sentences' in params:
            features.append(f"ANCHOR_TARGET={params['anchor_target_sentences']}")
        if params.get('min_sentences_per_chunk', 2) == 1:
            features.append("PRESERVES_ALL")
        feature_str = f" [{', '.join(features)}]" if features else ""
        config_list.append(f"    {i+1}. {name}{feature_str}")
        config_list.append(f"       {desc}")
    
    config_display = "\n".join(config_list)
    
    ap = argparse.ArgumentParser(
        description="Minimal semantic chunking controller (global splitter + anchor grouping)",
        epilog=f"Available configurations:\n{config_display}\n\nKey Points:\n  • Pure sentence-count control (no character limits)\n  • Contiguous or non‑contiguous semantic options\n  • Optional approximate sentence target & anchor size\n  • Advanced text cleaning for better sentence detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", required=True, help="Input TSV path")
    ap.add_argument("-o", "--output-dir", default="training_datasets", help="Output directory")
    ap.add_argument("-w", "--workers", type=int, default=1, help="Parallel configs workers")
    ap.add_argument("-c", "--config-workers", type=int, default=1, help="Workers per config (process batches)")
    ap.add_argument("--embedding-model", default=COMMON_DEFAULTS["embedding_model"], help="SentenceTransformer model")
    ap.add_argument("--device", default=COMMON_DEFAULTS["device_preference"], help="cpu | cuda | dml")
    ap.add_argument("--quiet", action="store_true", help="Silence controller logs (still prints interactive prompts)")
    ap.add_argument("--log-level", default="info", help="Log level: debug|info|warn|error (default: info)")
    ap.add_argument("--enable-text-cleaning", action="store_true", default=True, 
                    help="Enable text cleaning for better SpaCy sentence segmentation (default: True)")
    ap.add_argument("--disable-text-cleaning", action="store_true", 
                    help="Disable text cleaning, use raw documents (overrides --enable-text-cleaning)")
    ap.add_argument("--export-chunk-map", action="store_true", help="Xuất thêm file *_chunk_map.tsv với chỉ số câu & thống kê similarity (nếu hỗ trợ)")
    ap.add_argument(
        "--configs",
        help="Comma separated config names or numbers to run (default all)",
    )
    args = ap.parse_args()

    # USER PARAMETERS TAKE PRIORITY - only update defaults if user explicitly provided values
    user_embedding_model = args.embedding_model
    user_device_preference = args.device.lower()
    
    # Determine text cleaning setting based on user flags
    enable_text_cleaning = args.enable_text_cleaning and not args.disable_text_cleaning
    
    global CONTROLLER_SILENT
    CONTROLLER_SILENT = args.quiet
    set_controller_log_level(args.log_level)
    clog(f"startup embedding_model={user_embedding_model} device={user_device_preference} text_cleaning={enable_text_cleaning}", 'info')

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # filter configs - support both names and numbers
    if args.configs:
        configs = []
        for item in args.configs.split(","):
            item = item.strip()
            if item.isdigit() and 1 <= int(item) <= len(RUN_CONFIGURATIONS):
                configs.append(RUN_CONFIGURATIONS[int(item) - 1])
            else:
                # Try to match by name
                matching = [c for c in RUN_CONFIGURATIONS if c["name"] == item]
                configs.extend(matching)
    else:
        configs = RUN_CONFIGURATIONS

    if not configs:
        clog("no configurations selected – exiting", 'warning')
        print("Available configurations:")
        for i, cfg in enumerate(RUN_CONFIGURATIONS, 1):
            print(f"  {i}. {cfg['name']} - {cfg.get('description', 'No description')}")
        return

    # pool cho configs - pass user parameters directly, don't modify COMMON_DEFAULTS
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(user_embedding_model, user_device_preference),
        ) as exe:
            futs = {
                exe.submit(
                    run_config, 
                    config=cfg, 
                    input_path=input_path, 
                    output_dir=out_dir, 
                    config_workers=args.config_workers,
                    batch_size=BATCH_SIZE,
                    embedding_model=user_embedding_model,
                    device_preference=user_device_preference,
                    enable_text_cleaning=enable_text_cleaning,
                    collect_metadata=args.export_chunk_map,
                ): cfg["name"]
                for cfg in configs
            }
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    clog(f"config worker error name={futs[fut]} msg={e}", 'error')
    else:
        for cfg in configs:
            clog(f"starting config name={cfg['name']}", 'info')
            run_config(
                config=cfg, 
                input_path=input_path, 
                output_dir=out_dir, 
                config_workers=args.config_workers,
                batch_size=BATCH_SIZE,
                embedding_model=user_embedding_model,
                device_preference=user_device_preference,
                enable_text_cleaning=enable_text_cleaning,
                collect_metadata=args.export_chunk_map,
            )

    clog("all configurations completed", 'info')


if __name__ == "__main__":
    # Nếu được truyền tham số dòng lệnh thì dùng argparse, ngược lại mở interactive UI.
    import sys

    if len(sys.argv) > 1:
        mp.set_start_method("spawn", force=True)
        main()
    else:
        interactive_ui()