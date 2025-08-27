from __future__ import annotations

## ==== PYTHONPATH ADJUSTMENT ==================================================
# Thêm parent directory vào sys.path để import module Method/* khi chạy trực tiếp.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

## ==== GLOBAL ENV / LOG NOISE SUPPRESSION =====================================
# Giảm cảnh báo & log không cần thiết để output sạch.
import os, warnings, logging

# Tắt parallelism tokenizer (tránh cảnh báo + fork overhead) – phải trước import SentenceTransformer.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

## NOTE: csv.field_size_limit điều chỉnh ngay bên dưới khi cần đọc trường rất dài.

# Với phiên bản tokenizers mới: tắt parallelism qua API (an toàn nếu không hỗ trợ).
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

# Giảm mức log của sentence_transformers / transformers về ERROR.
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
from Method.Semantic_Grouping_Optimized import (
    semantic_chunk_passage_from_grouping_logic as semantic_grouping,
)
from Method.Semantic_Splitter_Optimized import (
    chunk_passage_text_splitter as semantic_splitter,
)
from Method.Text_Splitter_Char_Naive import chunk_passage_text_splitter as chunk_passage_text_splitter_char_naive
from Tool.rank_chunks_optimized import rank_and_filter_chunks_optimized

# ==== Default constants ====
BATCH_SIZE = 600  # dòng/đợt đọc pandas
COMMON_DEFAULTS = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "device_preference": "dml",  # "cuda", "dml", hoặc "cpu"
    "enable_text_cleaning": True,  # Enable text cleaning by default
}

# ==== Mapping helpers (integrated from data_process/file_mapping.py) ====
def _mapping_parse_topics(topics_file_path: str) -> Dict[str, str]:
    """Parse TREC topics file into {query_id: query_text} using desc + narr."""
    import re
    topics_data: Dict[str, str] = {}
    with open(topics_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    topic_entries = re.findall(r"<top>(.*?)</top>", content, re.DOTALL)
    for entry in topic_entries:
        num_match = re.search(r"<num>\s*Number:\s*(\d+)", entry)
        desc_match = re.search(r"<desc>\s*Description:(.*?)(?=<narr>|\Z)", entry, re.DOTALL)
        narr_match = re.search(r"<narr>\s*Narrative:(.*?)\Z", entry, re.DOTALL)
        if num_match:
            qid = num_match.group(1).strip()
            desc = desc_match.group(1).strip() if desc_match else ""
            narr = narr_match.group(1).strip() if narr_match else ""
            query_text = (desc + ". " + narr).replace('\n', ' ').replace('\r', '').replace('\t', ' ').strip()
            query_text = re.sub(r'\s+', ' ', query_text)
            topics_data[qid] = query_text
    return topics_data

def _mapping_add_query_text_to_tsv(file_final_path: str, file_topics_path: str, output_path: str) -> bool:
    """Add query_text to a TSV with columns: query_id, chunk_text, label."""
    topics = _mapping_parse_topics(file_topics_path)
    if not topics:
        clog("mapping: no queries parsed from topics", 'warning')
        return False
    processed_lines = 0
    skipped_lines = 0
    with open(file_final_path, 'r', encoding='utf-8') as src, open(output_path, 'w', encoding='utf-8') as out:
        # Read and ignore original header
        header = src.readline()
        out.write("query_text\tchunk_text\tlabel\n")
        line_num = 1
        for line in src:
            line_num += 1
            items = line.rstrip('\n').split('\t')
            if len(items) != 3:
                skipped_lines += 1
                continue
            query_id, chunk_text, label = items
            query_text = topics.get(query_id, "")
            if not query_text:
                skipped_lines += 1
                continue
            out.write(f"{query_text}\t{chunk_text}\t{label}\n")
            processed_lines += 1
    clog(f"mapping: processed={processed_lines} skipped={skipped_lines} out={output_path}", 'info')
    return True

"""
    
_MODEL_PRESETS = {
    # Retrieval-oriented (cao chất lượng, có thể mượt cục bộ):
    "1": ("thenlper/gte-large", "High Quality retrieval (≥4.5GB VRAM)"),
    "2": ("thenlper/gte-base", "Balanced retrieval (≈2GB VRAM)"),
    "6": ("BAAI/bge-large-en-v1.5", "High Quality retrieval (≈5GB VRAM)"),
    "7": ("BAAI/bge-small-en-v1.5", "Lightweight retrieval (≈0.3GB VRAM)"),
    "8": ("intfloat/e5-large-v2", "Retrieval (≈3.5GB VRAM)"),

    # Splitter-friendly / general-purpose SBERT:
    "3": ("sentence-transformers/all-MiniLM-L12-v2", "Fast general (≈0.7GB VRAM)"),
    "4": ("sentence-transformers/all-MiniLM-L6-v2", "Ultra-Fast general (≈0.5GB VRAM)"),
    "9": ("sentence-transformers/all-mpnet-base-v2", "Strong general-purpose (≈1.0GB VRAM)"),

    # Multilingual options:
    "10": ("paraphrase-multilingual-mpnet-base-v2", "Multilingual strong (≈1.0GB VRAM)"),
    "11": ("distiluse-base-multilingual-cased-v2", "Multilingual fast (≈0.6GB VRAM)"),

    # Auto suggestion:
    "5": ("auto", "Auto-select based on sys RAM/GPU"),
}

"""

# ==== Ranking defaults and hardcoded parameters (user-editable) ====
RANKING_DEFAULTS = {
    "model": "thenlper/gte-base",
    "workers": 4,
    "chunk_size": 50000,
    "upper_percentile": 80,
    "lower_percentile": 20,
    "filter_mode": "percentile",  # "percentile" or "fixed"
    "pos_sim_thr": 0.6,
    "neg_sim_thr": 0.4,
}

# When --hardcode flag is used, these parameters will be applied
HARDCODE_RANKING = {
    #"model": "thenlper/gte-base",
    "model" : "sentence-transformers/all-mpnet-base-v2",
    "device_preference": "dml",
    "workers": 4,
    "chunk_size": 50000,
    "upper_percentile": 80,
    "lower_percentile": 20,
    "filter_mode": "percentile",
    "pos_sim_thr": 0.6,
    "neg_sim_thr": 0.4,
}

# ==== Controller hardcoded run parameters (input/output, heatmap etc.) ====
HARD_CODED_CONTROLLER = {
    # Path to input TSV (query_id\tquery_text\tdocument_id\tdocument\tlabel)
    "input_path": str("F:/SematicSearch/integrated_robust04_modified_v2_subset_100rows.tsv"),
    # Output directory for chunk and rank artifacts
    "output_dir": str(Path("training_datasets")),
    # Controller execution knobs
    "workers": 1,             # parallel configs
    "config_workers": 1,      # workers per config (batch processing)
    "embedding_model": HARDCODE_RANKING["model"],
    "device": HARDCODE_RANKING["device_preference"],
    "enable_text_cleaning": True,
    # Allow running ranking step from controller (global gate)
    "allow_ranking": True,
    # Allow mapping step (add query_text from topics to filtered 3-col TSV)
    "allow_mapping": True,
    # Topics file path for mapping
    "topics_file": str(Path("F:/SematicSearch/robustirdata/topics.robust04.txt")),
    # Metadata & heatmap exports
    "export_chunk_map": True,     # write *_chunk_map.tsv
    "export_correlation": False,   # write heatmaps per document
    # Where to save heatmaps; default below if None
    "correlation_dir": str(Path("training_datasets") / "correlations"),
    # Overlay ideal boundaries in heatmaps, if available
    #"ideal_bounds_dir": str("F:/SematicSearch/tideal_bounds"),  # e.g., str(Path("training_datasets") / "ideal_bounds")
    # Optional subset of configurations to run; None means ALL
    # Example: ["semantic_splitter_global", "semantic_grouping_cluster"]
    "configs": "semantic_splitter_global",
}

# ==== Text Cleaning for SpaCy Sentence Segmentation ========================

def preprocess_format(text: str) -> str:

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
    """Làm sạch văn bản (đặc thù Robust04) để giảm boundary noise trước sentencizer."""
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
    
    # 2b. Loại bỏ các dòng/đoạn metadata còn sót kiểu: """Language: Spanish Article Type:BFN
    #    Bao phủ cả trường hợp có/không có khoảng trắng sau dấu hai chấm và có ngoặc kép
    #    Ví dụ: Language: Spanish Article Type:BFN | "Language: Russian Article Type: CSO"
    text = re.sub(
        r'\s*["“”\']{0,3}\s*Language:\s*\w+\s+Article\s*Type:\s*[A-Za-z0-9\-]+\.?\s*',
        ' ',
        text,
        flags=re.IGNORECASE,
    )

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
    #text = re.sub(r'\b([A-Z]{5,})\b(?=\s+[a-z])', r'\1.', text)
    
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
    """Guardrail: nếu cleaning làm mất quá nhiều nội dung thì revert."""
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
    """Khởi tạo resource cho mỗi worker process (spaCy sentencizer + SentenceTransformer)."""
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
    export_correlation: bool = False,
    correlation_dir: Optional[str] = None,
    ideal_bounds_dir: Optional[str] = None,
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

        clog(f"single-thread device={device_obj}", 'debug')
        model = SentenceTransformer(embedding_model, device=device_obj)

    method_choice = config["method_choice"]
    params = dict(config["params"])  # shallow copy

    # map
    func_map = {"1": semantic_grouping, "2": semantic_splitter, "3": chunk_passage_text_splitter_char_naive}
    chunk_func = func_map[method_choice]

    import os

    pid = os.getpid()
    clog(f"[controller] pid={pid} batch={batch_idx} rows={len(batch_df)} start", 'debug')

    # ---- Diameter enforcement helpers (optional) ----
    def _l2_normalize_rows(arr):
        if np is None:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return arr / norms

    def _split_indices_by_diameter(sim_matrix, start: int, end: int, threshold: float) -> List[Tuple[int, int]]:
        # Segment is [start, end) indices
        length = end - start
        if length <= 1 or np is None:
            return [(start, end)]
        # compute diameter = max distance = 1 - min(sim) over off-diagonal in submatrix
        sub = sim_matrix[start:end, start:end]
        if length == 2:
            diam = 1.0 - float(sub[0, 1])
        else:
            mask = ~np.eye(length, dtype=bool)
            min_sim = float(sub[mask].min()) if sub[mask].size else 1.0
            diam = 1.0 - min_sim
        if diam <= threshold:
            return [(start, end)]
        # choose split at lowest adjacent similarity inside segment
        adj_sims = [float(sim_matrix[i, i+1]) for i in range(start, end-1)]
        if not adj_sims:
            return [(start, end)]
        rel_min_idx = int(np.argmin(np.array(adj_sims)))
        split_pos = start + rel_min_idx + 1
        left = _split_indices_by_diameter(sim_matrix, start, split_pos, threshold)
        right = _split_indices_by_diameter(sim_matrix, split_pos, end, threshold)
        return left + right

    def _enforce_diameter_on_tuples(doc_text: str, tuples_in: List[Tuple[str, str, Optional[str]]], threshold: float) -> List[Tuple[str, str, Optional[str]]]:
        if np is None:
            return tuples_in
        results: List[Tuple[str, str, Optional[str]]] = []
        for chunk_id, chunk_text, meta in tuples_in:
            sents = extract_sentences_spacy(chunk_text)
            if len(sents) <= 1:
                results.append((chunk_id, chunk_text, meta))
                continue
            try:
                # use global SentenceTransformer model instantiated above
                emb = model.encode(sents, batch_size=32, normalize_embeddings=True)
                if emb is None or (hasattr(emb, 'size') and emb.size == 0):
                    results.append((chunk_id, chunk_text, meta))
                    continue
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=float)
                emb = _l2_normalize_rows(emb)
                sim = emb @ emb.T
                spans = _split_indices_by_diameter(sim, 0, len(sents), threshold)
                if len(spans) <= 1:
                    results.append((chunk_id, chunk_text, meta))
                else:
                    for idx_new, (a, b) in enumerate(spans):
                        sub_text = " ".join(sents[a:b])
                        new_id = f"{chunk_id}_d{idx_new}"
                        results.append((new_id, sub_text, None))
            except Exception:
                # any error: keep original
                results.append((chunk_id, chunk_text, meta))
        return results

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
            interview_processed = preprocess_format(cleaned_passage)
            
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

        # Optional: save per-document correlation matrix (sentence cosine heatmap)
        try:
            if export_correlation and correlation_dir:
                sents_full = extract_sentences_spacy(passage)
                if isinstance(sents_full, list) and len(sents_full) > 1:
                    # Import matplotlib lazily with non-interactive backend
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    emb_full = model.encode(sents_full, batch_size=32, normalize_embeddings=True)
                    if isinstance(emb_full, list):
                        emb_full = np.array(emb_full, dtype=float)
                    if emb_full is not None and hasattr(emb_full, 'shape') and emb_full.shape[0] >= 2:
                        sim_full = emb_full @ emb_full.T
                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(sim_full, vmin=0.0, vmax=1.0, cmap='viridis', aspect='auto')
                        ax.set_title(f"Cosine similarity – doc {doc_orig}")
                        ax.set_xlabel("Sentence index")
                        ax.set_ylabel("Sentence index")
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        out_png_dir = Path(correlation_dir)
                        out_png_dir.mkdir(parents=True, exist_ok=True)
                        out_png = out_png_dir / f"{doc_orig}.png"
                        try:
                            fig.tight_layout()
                            fig.savefig(out_png, dpi=120)
                        finally:
                            plt.close(fig)
        except Exception as e:
            try:
                clog(f"signals export failed doc={doc_orig} err={e}", 'warning')
            except Exception:
                pass

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

        # Hard fallback: nếu phương pháp trả về rỗng, tạo 1 chunk bao toàn văn
        if not tuples:
            tuples = [(f"{doc_id}_fallback", passage, None)]

        # Natural chunks: bỏ cưỡng bức đường kính/độ dài – không hậu xử lý cohesion

        # Optional: overwrite correlation heatmap with red boundary lines (Splitter only)
        try:
            if export_correlation and correlation_dir and method_choice == "2":
                sents_full = extract_sentences_spacy(passage)
                if isinstance(sents_full, list) and len(sents_full) > 1:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    import numpy as _np
                    emb_full = model.encode(sents_full, batch_size=32, normalize_embeddings=True)
                    if isinstance(emb_full, list):
                        emb_full = np.array(emb_full, dtype=float)
                    if emb_full is not None and hasattr(emb_full, 'shape') and emb_full.shape[0] >= 2:
                        sim_full = emb_full @ emb_full.T
                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(sim_full, vmin=0.0, vmax=1.0, cmap='viridis', aspect='auto')
                        ax.set_title(f"Cosine similarity – doc {doc_orig}")
                        ax.set_xlabel("Sentence index")
                        ax.set_ylabel("Sentence index")
                        # derive contiguous counts per chunk from splitter outputs
                        counts = []
                        for _cid, _ctext, _ in tuples:
                            counts.append(len(extract_sentences_spacy(_ctext)))
                        pos = 0
                        for c in counts[:-1]:
                            pos += c
                            ax.axvline(pos-0.5, color='red', linewidth=1.2)
                            ax.axhline(pos-0.5, color='red', linewidth=1.2)
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        out_png_dir = Path(correlation_dir)
                        out_png_dir.mkdir(parents=True, exist_ok=True)
                        out_png = out_png_dir / f"{doc_orig}.png"
                        try:
                            fig.tight_layout()
                            fig.savefig(out_png, dpi=120)
                        finally:
                            plt.close(fig)

                        # ==== Signals 1D plots with boundary markers ====
                        # Compute adjacent similarities and window gaps consistent with splitter
                        def _mean_vec(start: int, end: int):
                            if start >= end:
                                return None
                            v = emb_full[start:end]
                            if v is None or len(v) == 0:
                                return None
                            m = _np.mean(v, axis=0)
                            nrm = float(_np.linalg.norm(m) + 1e-9)
                            return m / nrm
                        adj_sims = [float((emb_full[i] @ emb_full[i+1])) for i in range(len(sents_full)-1)]
                        # Sigmoid sharpening đã loại bỏ khỏi splitter chính; vẫn giữ ở đây nếu cần debug trực quan
                        adj_sims_sig = []
                        W = int(params.get("window_size", 4) or 4)
                        window_gaps = []
                        for i in range(len(sents_full)-1):
                            Ls = max(0, i - W + 1); Le = i + 1
                            Rs = i + 1; Re = min(len(sents_full), i + 1 + W)
                            ml = _mean_vec(Ls, Le); mr = _mean_vec(Rs, Re)
                            window_gaps.append(0.0 if (ml is None or mr is None) else float(1.0 - (ml @ mr)))
                        # z-score + sigmoid signals
                        sims_arr = _np.array(adj_sims, dtype=float)
                        mean_sim = float(_np.mean(sims_arr)); std_sim = float(_np.std(sims_arr) + 1e-9)
                        # Triplet strength chỉ phục vụ visualization legacy
                        tau = 0.05
                        strength = []
                        base_for_strength = adj_sims_sig if adj_sims_sig else adj_sims
                        for i, _sim in enumerate(base_for_strength):
                            # compute dec_sum and inc_sum around i
                            dec_sum = 0.0
                            t = i
                            while t > 0 and base_for_strength[t] <= base_for_strength[t-1]:
                                dec_sum += max(0.0, base_for_strength[t-1] - base_for_strength[t])
                                t -= 1
                            inc_sum = 0.0
                            t = i
                            while t + 1 < len(base_for_strength) and base_for_strength[t+1] >= base_for_strength[t]:
                                inc_sum += max(0.0, base_for_strength[t+1] - base_for_strength[t])
                                t += 1
                            strength.append(dec_sum + inc_sum)
                        strength_sig = [1.0/(1.0+_np.exp(-(s/max(tau,1e-9)))) for s in strength]
                        # Boundaries indices from tuples
                        boundaries = []
                        acc = 0
                        for cc in counts[:-1]:
                            acc += cc
                            boundaries.append(acc)
                        # Prepare threshold BEFORE plotting
                        score_arr = _np.array([1.0/(1.0+_np.exp(-(s/max(tau,1e-9)))) for s in strength], dtype=float)
                        gamma = 0.5
                        med = float(_np.median(score_arr))
                        mad = float(_np.median(_np.abs(score_arr - med)) + 1e-9)
                        floor_sig = med + gamma * mad
                        # Plot signals
                        x = list(range(len(adj_sims)))
                        # Chuẩn bị boundary theo 2 phương pháp để overlay riêng
                        try:
                            from Method.Semantic_Splitter_Optimized import _c99_boundaries as _c99_b, _valley_boundaries as _valley_b
                        except Exception:
                            _c99_b = None; _valley_b = None
                        # Valley signal (z-score + sigmoid tuỳ chọn)
                        adj_for_valley = adj_sims
                        try:
                            sim_sigmoid_tau = params.get('sim_sigmoid_tau', None)
                            if sim_sigmoid_tau is not None:
                                tau_f = max(float(sim_sigmoid_tau), 1e-9)
                                arrv = _np.array(adj_sims, dtype=float)
                                mu_v = float(_np.mean(arrv)); sd_v = float(_np.std(arrv) + 1e-9)
                                z_v = (arrv - mu_v) / sd_v
                                adj_for_valley = (1.0/(1.0+_np.exp(-(z_v / tau_f)))).tolist()
                        except Exception:
                            adj_for_valley = adj_sims
                        min_spacing = int(params.get('min_boundary_spacing', 2) or 2)
                        min_first = int(params.get('min_first_boundary_index', 3) or 3)
                        valley_tau = float(params.get('valley_tau', 0.12) or 0.12)
                        valley_bounds_plot = []
                        c99_bounds_plot = []
                        if _valley_b is not None:
                            try:
                                valley_bounds_plot = _valley_b(adj_for_valley, triplet_tau=valley_tau, min_boundary_spacing=min_spacing, min_first_boundary_index=min_first)
                            except Exception:
                                valley_bounds_plot = []
                        if _c99_b is not None:
                            try:
                                c99_min_chunk = max(3, min_spacing)
                                c99_bounds_plot = _c99_b(emb_full, min_chunk_size=c99_min_chunk, max_cuts=None)
                            except Exception:
                                c99_bounds_plot = []

                        fig2, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
                        # Panel 1: adj sims (raw + optional sigmoid)
                        axs[0].plot(x, adj_sims, label='adj_sim_raw', color='#1f77b4', alpha=0.85)
                        if adj_sims_sig:
                            axs[0].plot(x, adj_sims_sig, label='adj_sim_sigmoid', color='#ff7f0e', alpha=0.9)
                        # Panel 2: strength (raw)
                        axs[1].plot(x, strength, label='strength', color='#111111')
                        # Panel 3: Valley signal + boundaries
                        axs[2].plot(x, adj_for_valley, label='valley_signal', color='#9467bd', alpha=0.9)
                        for b in valley_bounds_plot:
                            e = b - 1
                            axs[2].axvline(e, color='#9467bd', linestyle='-', linewidth=1.6, alpha=0.95)
                        # Panel 4: C99 overlay boundaries (dùng adj_sims nền mờ)
                        axs[3].plot(x, adj_sims, label='adj_sim_bg', color='#bdbdbd', alpha=0.5)
                        for b in c99_bounds_plot:
                            e = b - 1
                            axs[3].axvline(e, color='#1f77b4', linestyle='-', linewidth=1.6, alpha=0.95)
                        # Labels & legends
                        axs[0].set_ylabel('sim')
                        axs[1].set_ylabel('strength (raw)')
                        axs[2].set_ylabel('valley')
                        axs[3].set_ylabel('c99')
                        for ax in axs:
                            ax.legend(loc='upper right', fontsize=8, frameon=False)
                        eps = 1e-6
                        fs = max(eps, min(1.0 - eps, float(floor_sig)))
                        floor_raw = tau * _np.log(fs/(1.0-fs))
                        axs[1].axhline(floor_raw, color='orange', linestyle=':', linewidth=1)
                        axs[-1].set_xlabel('edge index (between i and i+1)')
                        for b in boundaries:
                            e = b-1
                            for ax in axs:
                                ax.axvline(e, color='#d62728', linestyle='--', linewidth=1.2)
                        # Overlay ideal boundaries if provided
                        try:
                            if ideal_bounds_dir:
                                from pathlib import Path as _P
                                gt_file = _P(ideal_bounds_dir) / f"{doc_orig}.bounds"
                                if gt_file.exists():
                                    txt = gt_file.read_text(encoding='utf-8').strip()
                                    if txt:
                                        ideal_edges = []
                                        for tok in txt.replace('\n', ',').split(','):
                                            tok = tok.strip()
                                            if tok.isdigit():
                                                ideal_edges.append(int(tok))
                                        for ib in ideal_edges:
                                            e = ib - 1
                                            for ax in axs:
                                                ax.axvline(e, color='green', linestyle='-', linewidth=1.2, alpha=0.9)
                        except Exception:
                            pass
                        # ----- Peak analysis & annotation -----
                        nms_w = int(params.get('min_boundary_spacing', 2) or 2)
                        # Detect local peaks
                        peaks = []  # (idx, score)
                        for i in range(1, len(score_arr)-1):
                            if score_arr[i] >= score_arr[i-1] and score_arr[i] >= score_arr[i+1]:
                                peaks.append((i, float(score_arr[i])))
                        # NMS simulation
                        peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                        selected_idx = []
                        reasons = {}
                        for idx, sc in peaks_sorted:
                            if sc < floor_sig:
                                reasons[idx] = 'FLOOR'
                                continue
                            if all(abs(idx - s) >= nms_w for s in selected_idx):
                                selected_idx.append(idx)
                                reasons[idx] = 'SELECTED'
                            else:
                                reasons[idx] = 'NMS'
                        # Plot annotations for peaks (raw only)
                        for idx, sc in peaks:
                            y = strength[idx]
                            color = 'red' if reasons.get(idx) == 'SELECTED' else ('gray' if reasons.get(idx) == 'NMS' else 'orange')
                            axs[1].scatter([idx], [y], color=color, s=16, zorder=3)
                            axs[1].annotate(f"{strength[idx]:.2f}", (idx, y), textcoords='offset points', xytext=(0, -10), ha='center', fontsize=7, color=color)
                        fig2.tight_layout()
                        out_sig = Path(correlation_dir) / f"{doc_orig}_signals.png"
                        fig2.savefig(out_sig, dpi=120)
                        plt.close(fig2)

                        # (disabled) per-boundary triplet plot sim-1/sim/sim+1
        except Exception:
            pass

        # Optional: grouping 1D color strip (Grouping only)
        try:
            if export_correlation and correlation_dir and method_choice == "1":
                sents_full = extract_sentences_spacy(passage)
                if isinstance(sents_full, list) and len(sents_full) >= 1:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    import numpy as _np
                    import json as _json

                    n_sent = len(sents_full)
                    # Build assignment per sentence from metadata if available
                    assign = _np.full((n_sent,), -1, dtype=int)
                    cluster_count = 0
                    for cid, (_cid, _ctext, _meta) in enumerate(tuples):
                        if not _meta:
                            continue
                        try:
                            md = _json.loads(_meta)
                        except Exception:
                            continue
                        idx_str = md.get("sent_indices", "")
                        if not idx_str:
                            continue
                        for tok in str(idx_str).split(","):
                            tok = tok.strip()
                            if tok.isdigit():
                                si = int(tok)
                                if 0 <= si < n_sent:
                                    assign[si] = cid
                                    cluster_count = max(cluster_count, cid + 1)

                    # If no metadata assignments found, skip plotting to avoid misleading figure
                    if _np.any(assign >= 0):
                        # Map assignments to color indices: -1 -> 0 (grey), clusters -> 1..cluster_count
                        color_idx = assign.copy()
                        color_idx = color_idx + 1  # -1 -> 0, 0 -> 1, ...
                        # Build palette: first color grey for unassigned, then tab20 colors for clusters
                        from matplotlib.colors import ListedColormap
                        base = plt.get_cmap('tab20', max(cluster_count, 1))
                        palette = [(0.7, 0.7, 0.7, 1.0)] + [base(i) for i in range(max(cluster_count, 1))]
                        cmap = ListedColormap(palette)

                        fig, ax = plt.subplots(figsize=(max(6, n_sent * 0.12), 1.8))
                        ax.imshow(color_idx.reshape(1, -1), aspect='auto', cmap=cmap, vmin=0, vmax=max(cluster_count, 1))
                        # Thin black separators between sentences
                        for x in range(1, n_sent):
                            ax.axvline(x - 0.5, color='black', linewidth=0.5, alpha=0.5)
                        ax.set_yticks([])
                        # Add x-axis ticks as sentence indices (1-based), limit to ~20 ticks for readability
                        try:
                            step = max(1, int(round(n_sent / 20.0)))
                        except Exception:
                            step = max(1, n_sent // 20)
                        ticks = list(range(0, n_sent, step))
                        if (n_sent - 1) not in ticks:
                            ticks.append(n_sent - 1)
                        labels = [str(t + 1) for t in ticks]
                        ax.set_xticks(ticks)
                        ax.set_xticklabels(labels, fontsize=8)
                        ax.set_xlabel('sentence index')
                        # Build legend: color ↔ cluster name (chunk_id)
                        try:
                            from matplotlib.patches import Patch
                            clusters_present = sorted(set(int(c) for c in assign.tolist() if int(c) >= 0))
                            legend_patches = []
                            legend_labels = []
                            for cid2 in clusters_present:
                                color = palette[cid2 + 1]
                                # Prefer chunk_id string from tuples
                                try:
                                    chunk_name = str(tuples[cid2][0])
                                    # Shorten label if too long: keep suffix after last '_' if looks like '..._clusterX'
                                    if 'cluster' in chunk_name:
                                        idx_us = chunk_name.rfind('_')
                                        if idx_us != -1 and idx_us + 1 < len(chunk_name):
                                            chunk_short = chunk_name[idx_us + 1:]
                                        else:
                                            chunk_short = chunk_name
                                    else:
                                        chunk_short = chunk_name
                                except Exception:
                                    chunk_short = f"cluster{cid2}"
                                legend_patches.append(Patch(facecolor=color, edgecolor='none'))
                                legend_labels.append(chunk_short)
                            if legend_patches:
                                ax.legend(
                                    handles=legend_patches,
                                    labels=legend_labels,
                                    loc='upper center',
                                    bbox_to_anchor=(0.5, 1.35),
                                    ncol=max(1, min(len(legend_labels), 6)),
                                    fontsize=8,
                                    frameon=False,
                                )
                        except Exception:
                            pass

                        # No title to avoid overlap with legend
                        out_strip = Path(correlation_dir) / f"{doc_orig}_grouping.png"
                        fig.tight_layout()
                        fig.savefig(out_strip, dpi=140)
                        plt.close(fig)
        except Exception:
            pass
        
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
                
                # Try to capture grouping engine used from metadata (if any)
                engine_used = None
                if method_choice == "1" and collect_metadata:
                    try:
                        import json as _json
                        for _cid, _ctext, _meta in tuples:
                            if _meta:
                                md = _json.loads(_meta)
                                eng = md.get('method_used')
                                if eng:
                                    engine_used = eng
                                    break
                    except Exception:
                        engine_used = None
                engine_info = f" engine={engine_used}" if engine_used else ""
                clog(f"doc_idx={local_idx} method={method_name}{engine_info} chunks={len(tuples)} size_avg={avg_size:.0f} min={min_size} max={max_size}{balanced_info}", 'info')
        
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
    export_correlation: bool = False,
    correlation_dir: Optional[Path] = None,
    ideal_bounds_dir: Optional[Path] = None,
    rank_after: bool = False,
    ranking_params: Optional[Dict[str, Any]] = None,
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

    # Announce effective settings for this config
    clog(
        f"config={config['name']} begin model={actual_embedding_model} device={actual_device_preference} clean={'on' if actual_enable_text_cleaning else 'off'} rank_after={rank_after}",
        'info'
    )

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
        # Create metadata writer if requested
        w_map = csv.writer(fmap, delimiter='\t') if collect_metadata else None
        if w_map:
            w_map.writerow(["query_id", "document_id", "chunk_id", "sent_indices", "n_sent", "sim_mean", "sim_min", "sim_max", "sim_std", "anchor", "anchor_centrality"])  # some fields may be blank

        # pool xử lý batch with streaming write
        if config_workers > 1:
            with ProcessPoolExecutor(
                max_workers=config_workers,
                initializer=_worker_init,
                initargs=(actual_embedding_model, actual_device_preference),
            ) as exe:
                futures = {
                    exe.submit(_process_batch, df, config, idx, actual_embedding_model, actual_device_preference, actual_enable_text_cleaning, max_chunk_chars, collect_metadata, export_correlation, str(correlation_dir) if correlation_dir else None, str(ideal_bounds_dir) if ideal_bounds_dir else None): idx 
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
                    batch_results = _process_batch(df, config, idx, actual_embedding_model, actual_device_preference, actual_enable_text_cleaning, max_chunk_chars, collect_metadata, export_correlation, str(correlation_dir) if correlation_dir else None, str(ideal_bounds_dir) if ideal_bounds_dir else None)
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

    # Optional: run ranking after chunking
    if rank_after:
        try:
            rp = dict(RANKING_DEFAULTS)
            if ranking_params:
                rp.update(ranking_params)
            clog(
                f"ranking start model={rp['model']} mode={rp['filter_mode']} up={rp['upper_percentile']} low={rp['lower_percentile']} workers={rp['workers']} chunks_in={outfile}",
                'info'
            )
            ranked_path = rank_and_filter_chunks_optimized(
                chunks_tsv=str(outfile),
                output_dir=output_dir,
                original_tsv=str(input_path),
                upper_percentile=int(rp.get('upper_percentile', 80)),
                lower_percentile=int(rp.get('lower_percentile', 20)),
                model_name=str(rp.get('model', RANKING_DEFAULTS['model'])),
                max_workers=int(rp.get('workers', 1)),
                chunk_size=int(rp.get('chunk_size', 50000)),
                pos_sim_thr=float(rp.get('pos_sim_thr', 0.6)),
                neg_sim_thr=float(rp.get('neg_sim_thr', 0.4)),
                filter_mode=str(rp.get('filter_mode', 'percentile')),
            )
            clog(f"ranking completed file={ranked_path}", 'info')
        except Exception as e:
            clog(f"ranking failed: {e}", 'error')

        # Optional mapping step: convert query_id to query_text using topics file
        try:
            hc = HARD_CODED_CONTROLLER
            allow_map = bool(hc.get("allow_mapping", False))
            topics_file = hc.get("topics_file")
            if allow_map and topics_file and ranked_path:
                ranked_path_str = str(ranked_path)
                # Prefer the 3-col file output (same name) already saved by ranker
                # If needed, map on ranked_path to produce *_with_querytext.tsv
                mapped_out = str(Path(ranked_path_str).with_name(Path(ranked_path_str).stem + "_with_querytext.tsv"))
                ok = _mapping_add_query_text_to_tsv(ranked_path_str, str(topics_file), mapped_out)
                if ok:
                    clog(f"mapping completed file={mapped_out}", 'info')
                else:
                    clog("mapping skipped (no queries parsed or no matches)", 'warning')
        except Exception as e:
            clog(f"mapping failed: {e}", 'error')
    clog(f"stats chunks={total_chunks} elapsed={elapsed:.1f}s rate={chunks_per_sec:.1f}/s avg_per_doc={avg_chunks_per_doc:.2f}{split_info}{preserve_info}", 'info')


# ==== RUN_CONFIGURATIONS =====================================================


# ==== CONFIGURATIONS (moved near the top for easy editing) ====================
RUN_CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "name": "semantic_splitter_global",
        "method_choice": "2",
        "params": {
            # C99-based splitter params (no fixed size constraints)
            "min_boundary_spacing": 12,
            "min_first_boundary_index": 8,
            "min_chunk_len": 8,      # optional post-process merge threshold
            "max_chunk_len": 24,     # optional post-process split threshold
            # Hybrid params
            "hybrid_mode": "intersection",  # or "union" | "union_weighted"
            "valley_tau": 0.1,
            "sim_sigmoid_tau": 0.45,
            "vote_thr": 0.9,
            # C99 advanced
            "c99_use_local_rank": False,
            "c99_mask_size": 11,
            "c99_stopping": "gain",
            "c99_knee_c": 1.3,
            "c99_smooth_window": 5,
            "refine_max_shift": 2,
            # New robustness knobs
            "smooth_adj_window": 5,
            "reassign_k": 2,
            "reassign_margin": 0.03,
            "dp_local_adjust": True,
            "dp_window": 6,
            "dp_improve_eps": 0.001,
            # Multi-pass (disabled by default here)
            "multi_pass_dp": True,
            "dp_global_penalty": 0.6,
            "dp_min_first_boundary_index": 8,
            "multi_pass_short_len": 3,
        },
        "description": "Contiguous semantic splitter using C99 over embedding sim matrix (+ optional length post-process)"
    },
    {
        "name": "semantic_grouping_cluster",
        "method_choice": "1",
        "params": {
            "engine": "spectral",  # 'rmt' (default, with fallback) or 'spectral' (spectral only)
            # Sharpening nhẹ để tăng độ tương phản
            "sigmoid_tau_group": 0.12,
            # Làm đồ thị k‑NN thưa hơn và ngưỡng cạnh cao hơn để tách cụm mạnh hơn
            "knn_k": 6,
            "edge_floor": 0.50,
            # Cho phép auto‑K lớn hơn để lấy nhiều cụm nếu eigengap ủng hộ
            "spectral_kmax": 12,
            # Ép tách cụm lớn sớm hơn
            "cap_soft": 24,
            # Hạn chế hợp nhất: xem cụm nhỏ là "đủ lớn" sớm và yêu cầu độ tương đồng cao mới hợp nhất
            "small_group_min": 6,
            "tau_merge": 0.22,
            # Giảm tái gán ranh giới tùy tiện
            "reassign_delta": 0.02,
            # Tham số RMT vẫn giữ để có thể chuyển engine nhanh nếu cần
            "rmt_keep_eigs": 2,
            "mod_gamma_start": 0.5,
            "mod_gamma_end": 1.2,
            "mod_gamma_step": 0.15
        },
        "description": "RMT + multiscale modularity (primary) with Spectral fallback; split/merge + one‑pass reassignment"
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
    # Retrieval-oriented (cao chất lượng, có thể mượt cục bộ):
    "1": ("thenlper/gte-large", "High Quality retrieval (≥4.5GB VRAM)"),
    "2": ("thenlper/gte-base", "Balanced retrieval (≈2GB VRAM)"),
    "6": ("BAAI/bge-large-en-v1.5", "High Quality retrieval (≈5GB VRAM)"),
    "7": ("BAAI/bge-small-en-v1.5", "Lightweight retrieval (≈0.3GB VRAM)"),
    "8": ("intfloat/e5-large-v2", "Retrieval (≈3.5GB VRAM)"),

    # Splitter-friendly / general-purpose SBERT:
    "3": ("sentence-transformers/all-MiniLM-L12-v2", "Fast general (≈0.7GB VRAM)"),
    "4": ("sentence-transformers/all-MiniLM-L6-v2", "Ultra-Fast general (≈0.5GB VRAM)"),
    "9": ("sentence-transformers/all-mpnet-base-v2", "Strong general-purpose (≈1.0GB VRAM)"),

    # Multilingual options:
    "10": ("paraphrase-multilingual-mpnet-base-v2", "Multilingual strong (≈1.0GB VRAM)"),
    "11": ("distiluse-base-multilingual-cased-v2", "Multilingual fast (≈0.6GB VRAM)"),

    # Auto suggestion:
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
        if 'anchor_similarity_floor' in params:
            params_info.append(f"floor={params['anchor_similarity_floor']}")
        if params.get('enforce_diameter'):
            params_info.append(f"diameter≤{params.get('diameter_threshold', 0.6)}")
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

    # ----- Export chunk map (metadata) option -----
    export_map_ans = input("Export chunk map with sentence indices & similarity stats? (y/N): ").strip().lower()
    export_chunk_map = export_map_ans == 'y'
    export_corr_ans = input("Export per-document correlation (cosine heatmap) images? (y/N): ").strip().lower()
    export_corr = export_corr_ans == 'y'
    corr_default = Path(HARD_CODED_CONTROLLER["correlation_dir"]) if HARD_CODED_CONTROLLER.get("correlation_dir") else (out_dir / "correlations")
    corr_dir = corr_default
    if export_corr:
        user_corr = input(f"Correlation output dir [{corr_default}]: ").strip()
        corr_dir = Path(user_corr) if user_corr else corr_default
        corr_dir.mkdir(exist_ok=True)
        # Optional: ideal boundaries overlay
        ideal_ans = input("Overlay ideal boundaries on signals plots from files *.bounds? (y/N): ").strip().lower()
        if ideal_ans == 'y':
            ideal_default = Path(HARD_CODED_CONTROLLER["ideal_bounds_dir"]) if HARD_CODED_CONTROLLER.get("ideal_bounds_dir") else (out_dir / "ideal_bounds")
            ideal_dir_input = input(f"Enter directory containing {{document_id}}.bounds files [{ideal_default}]: ").strip()
            ideal_dir = Path(ideal_dir_input or ideal_default)
            ideal_dir.mkdir(exist_ok=True)
        else:
            ideal_dir = None

    print("\n=== SUMMARY ===")
    print(f"Input        : {input_path}")
    print(f"Output dir   : {out_dir}")
    print(f"Embedding    : {current_embedding_model}")
    print(f"Configurations: {len(selected_cfgs)} selected")
    for cfg in selected_cfgs:
        params_summary = []
        params = cfg.get('params', {})
        if 'anchor_similarity_floor' in params:
            params_summary.append(f"floor={params['anchor_similarity_floor']}")
        if params.get('enforce_diameter'):
            params_summary.append(f"diameter≤{params.get('diameter_threshold', 0.6)}")
        param_str = f" ({', '.join(params_summary)})" if params_summary else ""
        print(f"  - {cfg['name']}{param_str}")
    print(f"Workers      : per-config={cfg_workers} max_chunk_chars={max_chunk_chars}\n")
    print(f"Export map   : {'YES' if export_chunk_map else 'no'}\n")

    # ----- Optional ranking after chunking -----
    allow_rank_default = bool(HARD_CODED_CONTROLLER.get("allow_ranking", True))
    if not allow_rank_default:
        print("Ranking is disabled by HARD_CODED_CONTROLLER.allow_ranking=False")
        do_rank = False
    else:
        do_rank = input("Run ranking after chunking? (y/N): ").strip().lower() == 'y'
    rank_params = None
    if do_rank:
        use_hard = input("Use HARDCODE ranking params? (y/N): ").strip().lower() == 'y'
        if use_hard:
            rank_params = dict(HARDCODE_RANKING)
            print(f"Using HARDCODE params: {rank_params}")
        else:
            # Collect minimal interactive ranking params
            model = input(f"Ranking model [{RANKING_DEFAULTS['model']}]: ").strip() or RANKING_DEFAULTS['model']
            workers = int(input(f"Ranking workers [{RANKING_DEFAULTS['workers']}]: ") or str(RANKING_DEFAULTS['workers']))
            chunk_sz = int(input(f"Ranking chunk size [{RANKING_DEFAULTS['chunk_size']}]: ") or str(RANKING_DEFAULTS['chunk_size']))
            mode = input("Filter mode [percentile|fixed] (default percentile): ").strip() or RANKING_DEFAULTS['filter_mode']
            up = int(input(f"Upper percentile POS [{RANKING_DEFAULTS['upper_percentile']}]: ") or str(RANKING_DEFAULTS['upper_percentile']))
            low = int(input(f"Lower percentile NEG [{RANKING_DEFAULTS['lower_percentile']}]: ") or str(RANKING_DEFAULTS['lower_percentile']))
            pos_thr = float(input(f"Positive cosine threshold [{RANKING_DEFAULTS['pos_sim_thr']}]: ") or str(RANKING_DEFAULTS['pos_sim_thr']))
            neg_thr = float(input(f"Negative cosine threshold [{RANKING_DEFAULTS['neg_sim_thr']}]: ") or str(RANKING_DEFAULTS['neg_sim_thr']))
            rank_params = {
                "model": model,
                "workers": workers,
                "chunk_size": chunk_sz,
                "filter_mode": mode,
                "upper_percentile": up,
                "lower_percentile": low,
                "pos_sim_thr": pos_thr,
                "neg_sim_thr": neg_thr,
            }

    # ----- Optional mapping after ranking -----
    do_map = False
    map_topics = None
    if do_rank:
        allow_map_default = bool(HARD_CODED_CONTROLLER.get("allow_mapping", False))
        do_map = input(f"Run mapping (add query_text) after ranking? (y/N) [default {'Y' if allow_map_default else 'N'}]: ").strip().lower() == 'y' if allow_map_default else (input("Run mapping (add query_text) after ranking? (y/N): ").strip().lower() == 'y')
        if do_map:
            map_topics_default = HARD_CODED_CONTROLLER.get("topics_file") or (Path("robustirdata") / "topics.robust04.txt")
            map_topics_in = input(f"Path to topics file [{map_topics_default}]: ").strip()
            map_topics = Path(map_topics_in) if map_topics_in else Path(map_topics_default)
            # Update hardcoded so run_config sees it
            HARD_CODED_CONTROLLER["allow_mapping"] = True
            HARD_CODED_CONTROLLER["topics_file"] = str(map_topics)

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
            collect_metadata=export_chunk_map,
            export_correlation=export_corr,
            correlation_dir=corr_dir,
            ideal_bounds_dir=ideal_dir,
            rank_after=do_rank,
            ranking_params=rank_params,
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
        if 'anchor_similarity_floor' in params:
            features.append(f"FLOOR={params['anchor_similarity_floor']}")
        if params.get('enforce_diameter'):
            features.append(f"DIAM≤{params.get('diameter_threshold', 0.6)}")
        feature_str = f" [{', '.join(features)}]" if features else ""
        config_list.append(f"    {i+1}. {name}{feature_str}")
        config_list.append(f"       {desc}")
    
    config_display = "\n".join(config_list)
    
    ap = argparse.ArgumentParser(
        description="Minimal semantic chunking controller (global splitter + anchor grouping)",
        epilog=f"Available configurations:\n{config_display}\n\nKey Points:\n  • Pure sentence-count control (no character limits)\n  • Contiguous or non‑contiguous semantic options\n  • Optional approximate sentence target & anchor size\n  • Advanced text cleaning for better sentence detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", required=False, help="Input TSV path (required unless --use-hardcoded)")
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
    ap.add_argument("--export-correlation", action="store_true", help="Xuất ảnh heatmap cosine similarity per-document")
    ap.add_argument("--ideal-bounds-dir", help="Thư mục chứa các file {document_id}.bounds (danh sách edge theo dòng hoặc dấu phẩy) để overlay lên signals")
    # Ranking options
    ap.add_argument("--rank-after", action="store_true", help="Run ranking after chunking and write filtered outputs")
    ap.add_argument("--hardcode", action="store_true", help="Use HARDCODE_RANKING parameters for ranking")
    ap.add_argument("--rank-model", help="Ranking model name (overrides defaults unless --hardcode)")
    ap.add_argument("--rank-workers", type=int, help="Ranking workers (overrides defaults unless --hardcode)")
    ap.add_argument("--rank-chunk-size", type=int, help="Ranking chunk size (overrides defaults unless --hardcode)")
    ap.add_argument("--rank-filter-mode", choices=["percentile", "fixed"], help="Ranking filter mode")
    ap.add_argument("--rank-up", type=int, help="Upper percentile for positives")
    ap.add_argument("--rank-low", type=int, help="Lower percentile for negatives")
    ap.add_argument("--rank-pos-thr", type=float, help="Positive cosine threshold (fixed mode)")
    ap.add_argument("--rank-neg-thr", type=float, help="Negative cosine threshold (fixed mode)")
    # Mapping options
    ap.add_argument("--map-after", action="store_true", help="Run mapping (add query_text) after ranking")
    ap.add_argument("--topics-file", help="Path to topics file for mapping")
    ap.add_argument("--grouping-engine", choices=["rmt", "spectral"], help="Engine cho semantic_grouping: rmt (mặc định, có fallback) hoặc spectral (chỉ spectral)")
    ap.add_argument(
        "--configs",
        help="Comma separated config names or numbers to run (default all)",
    )
    # Controller hardcoded toggle (with alias for convenience)
    ap.add_argument("--use-hardcoded", action="store_true", dest="use_hardcoded", help="Use HARD_CODED_CONTROLLER for input/output and heatmap options")
    ap.add_argument("--use-hardcode", action="store_true", dest="use_hardcoded", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # USER PARAMETERS TAKE PRIORITY - only update defaults if user explicitly provided values
    user_embedding_model = args.embedding_model
    user_device_preference = args.device.lower()
    
    # Determine text cleaning setting based on user flags
    enable_text_cleaning = args.enable_text_cleaning and not args.disable_text_cleaning
    
    global CONTROLLER_SILENT
    CONTROLLER_SILENT = args.quiet
    set_controller_log_level(args.log_level)

    # Validate required input unless using hardcoded
    if not args.use_hardcoded and not args.input:
        ap.error("-i/--input is required unless --use-hardcoded is provided")

    # If --use-hardcoded is set, override input/output and export options
    if args.use_hardcoded:
        hc = HARD_CODED_CONTROLLER
        # Apply paths
        args.input = hc.get("input_path", args.input)
        args.output_dir = hc.get("output_dir", args.output_dir)
        # Apply controller execution knobs
        args.workers = hc.get("workers", args.workers)
        args.config_workers = hc.get("config_workers", args.config_workers)
        args.embedding_model = hc.get("embedding_model", args.embedding_model)
        args.device = hc.get("device", args.device)
        # Reflect hardcoded overrides into effective variables
        user_embedding_model = args.embedding_model
        user_device_preference = args.device.lower()
        # Apply text cleaning
        if hc.get("enable_text_cleaning", True):
            args.enable_text_cleaning = True
            args.disable_text_cleaning = False
        else:
            args.enable_text_cleaning = False
            args.disable_text_cleaning = True
        enable_text_cleaning = args.enable_text_cleaning and not args.disable_text_cleaning
        # Export options
        if hc.get("export_chunk_map", False):
            args.export_chunk_map = True
        if hc.get("export_correlation", False):
            args.export_correlation = True
        # Ideal bounds dir
        if hc.get("ideal_bounds_dir"):
            args.ideal_bounds_dir = hc.get("ideal_bounds_dir")
        # Prepare correlation dir
        correlation_dir_override = hc.get("correlation_dir")
        # Apply hardcoded configs selection if provided
        if hc.get("configs"):
            args.configs = hc.get("configs")
        # Honor mapping default from hardcoded
        if not hc.get("allow_mapping", False):
            args.map_after = False
        else:
            # Provide default topics file if not given
            if not getattr(args, 'topics_file', None):
                args.topics_file = hc.get("topics_file")
            # Auto-enable mapping when allowed in hardcoded mode
            if hc.get("topics_file"):
                args.map_after = True
        # Honor allow_ranking gate for CLI
        if not hc.get("allow_ranking", True):
            # Disable rank-after even if user passed it
            try:
                args.rank_after = False
            except Exception:
                pass
        else:
            # Auto-enable ranking with HARDCODE_RANKING if not specified
            if not getattr(args, 'rank_after', False):
                args.rank_after = True
                args.hardcode = True
                clog("hardcoded: auto-enable ranking with HARDCODE_RANKING", 'info')
    else:
        correlation_dir_override = None
    # Log effective startup settings AFTER applying overrides
    clog(
        f"startup embedding_model={user_embedding_model} device={user_device_preference} text_cleaning={enable_text_cleaning} use_hardcoded={args.use_hardcoded} rank_after={bool(args.rank_after)} map_after={bool(args.map_after)}",
        'info'
    )

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Ensure correlation directory exists if exporting correlations
    if args.export_correlation:
        try:
            corr_path = Path(correlation_dir_override) if correlation_dir_override else (out_dir / "correlations")
            corr_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            clog(f"failed to create correlation dir: {e}", 'warning')

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

    # Nếu user chỉ định grouping engine, ép vào các cấu hình grouping
    if args.grouping_engine:
        for cfg in configs:
            try:
                if cfg.get("method_choice") == "1":
                    cfg.setdefault("params", {})
                    cfg["params"]["engine"] = args.grouping_engine
            except Exception:
                pass

    # Build ranking params if requested
    do_rank_after = bool(args.rank_after)
    rank_params_cli = None
    if do_rank_after:
        if args.hardcode:
            rank_params_cli = dict(HARDCODE_RANKING)
        else:
            rp = dict(RANKING_DEFAULTS)
            if args.rank_model: rp['model'] = args.rank_model
            if args.rank_workers is not None: rp['workers'] = args.rank_workers
            if args.rank_chunk_size is not None: rp['chunk_size'] = args.rank_chunk_size
            if args.rank_filter_mode: rp['filter_mode'] = args.rank_filter_mode
            if args.rank_up is not None: rp['upper_percentile'] = args.rank_up
            if args.rank_low is not None: rp['lower_percentile'] = args.rank_low
            if args.rank_pos_thr is not None: rp['pos_sim_thr'] = args.rank_pos_thr
            if args.rank_neg_thr is not None: rp['neg_sim_thr'] = args.rank_neg_thr
            rank_params_cli = rp

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
                    export_correlation=args.export_correlation,
                    correlation_dir=Path(correlation_dir_override) if correlation_dir_override else out_dir / "correlations",
                    ideal_bounds_dir=Path(args.ideal_bounds_dir) if args.ideal_bounds_dir else None,
                    rank_after=do_rank_after,
                    ranking_params=rank_params_cli,
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
                export_correlation=args.export_correlation,
                correlation_dir=Path(correlation_dir_override) if correlation_dir_override else out_dir / "correlations",
                ideal_bounds_dir=Path(args.ideal_bounds_dir) if args.ideal_bounds_dir else None,
                rank_after=do_rank_after,
                ranking_params=rank_params_cli,
            )

    clog("all configurations completed", 'info')


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mp.set_start_method("spawn", force=True)
        main()
    else:
        interactive_ui()