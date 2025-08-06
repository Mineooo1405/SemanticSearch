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

import pandas as pd

# ==== Import optimized chunking methods ====
# Note: Semantic Splitter has been optimized - removed adaptive parameters (target_tokens, tolerance, enable_adaptive)
from Method.Semantic_Grouping_Optimized import (
    semantic_chunk_passage_from_grouping_logic as semantic_grouping,
)
from Method.Semantic_Splitter_Optimized import (
    chunk_passage_text_splitter as semantic_splitter,
)
from Method.Text_Splitter_Optimized import (
    chunk_passage_text_splitter as simple_splitter,
)


"""
# Enable text cleaning (default)
python simple_chunk_controller.py -i input.tsv -o output --configs semantic_grouping_balanced

# Disable text cleaning (raw documents)
python simple_chunk_controller.py -i input.tsv -o output --configs semantic_grouping_balanced --disable-text-cleaning
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

    print(f"[PID {os.getpid()}] Using device: {device_obj}")
    _GLOBALS["model"] = SentenceTransformer(model_name, device=device_obj)


# ==== Helper: chunk 1 batch ==================================================

def _process_batch(
    batch_df: pd.DataFrame,
    config: Dict[str, Any],
    batch_idx: int,
    embedding_model: str,
    device_preference: str,
    enable_text_cleaning: bool = True,
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
    func_map = {"1": semantic_grouping, "2": semantic_splitter, "3": simple_splitter}
    chunk_func = func_map[method_choice]

    import os

    pid = os.getpid()
    print(f"[PID {pid}]Processing batch {batch_idx} (rows={len(batch_df)})")

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
                print(f"Warning: Text cleaning may be too aggressive for doc {doc_orig}, using original")
                interview_processed = original_passage
            
            # Use preprocessed text for processing
            passage = interview_processed

        if not isinstance(passage, str) or not passage.strip():
            continue

        # TREC ROBUST04 FILTER: Skip documents marked as having no information
        if passage.strip() == "This document has no information.":
            print(f"[PID {pid}] Skipping document {doc_orig} - marked as having no information")
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

        tuples: List[Tuple[str, str, Optional[str]]] = chunk_func(
            doc_id=doc_id,
            passage_text=passage,
            **params,
            **extra,
            silent=True,
            output_dir=None,
        )
        for chunk_id, chunk_text, _ in tuples:
            # Output: query_id  | document_id | chunk_text | label
            results.append([query_id, doc_orig, chunk_text, label])
    print(f"[PID {pid}]Batch {batch_idx} done – {len(results)} chunks")
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
):
    """Chạy chunking cho 1 cấu hình."""
    import time

    # Use provided parameters or fall back to defaults - USER PARAMETERS TAKE PRIORITY
    actual_embedding_model = embedding_model if embedding_model is not None else COMMON_DEFAULTS["embedding_model"]
    actual_device_preference = device_preference if device_preference is not None else COMMON_DEFAULTS["device_preference"]
    actual_enable_text_cleaning = enable_text_cleaning if enable_text_cleaning is not None else COMMON_DEFAULTS["enable_text_cleaning"]

    start = time.time()
    outfile = output_dir / f"{config['name']}_chunks.tsv"

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

    # pool xử lý batch
    if config_workers > 1:
        with ProcessPoolExecutor(
            max_workers=config_workers,
            initializer=_worker_init,
            initargs=(actual_embedding_model, actual_device_preference),
        ) as exe:
            futures = {
                exe.submit(_process_batch, df, config, idx, actual_embedding_model, actual_device_preference, actual_enable_text_cleaning): idx 
                for idx, df in all_batches
            }
            batch_results = []
            for fut in as_completed(futures):
                batch_results.extend(fut.result())
    else:
        batch_results = []
        for idx, df in all_batches:
            batch_results.extend(_process_batch(df, config, idx, actual_embedding_model, actual_device_preference, actual_enable_text_cleaning))

    # ghi file
    outfile.parent.mkdir(exist_ok=True)
    with open(outfile, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "document_id", "chunk_text", "label"])
        w.writerows(batch_results)

    elapsed = time.time() - start
    print(f"[CONFIG DONE] {config['name']} → {outfile} | {len(batch_results)} chunks | {elapsed:.1f}s")


# ==== RUN_CONFIGURATIONS =====================================================


RUN_CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "name": "semantic_splitter_no_limit",
        "method_choice": "2",
        "params": {
            "chunk_size": None,  # No character limit
            "chunk_overlap": 0,
            "semantic_threshold": 0.35,
            "min_sentences_per_chunk": 3,
        },
        "description": "Semantic splitter chunking without size limits"
    },
    {
        "name": "semantic_grouping_no_limit",
        "method_choice": "1",
        "params": {
            "initial_threshold": 1,
            "decay_factor": 0.85,
            "min_threshold": 0.1, 
            "min_sentences_per_chunk": 2,  # Lowered to allow smaller but coherent chunks
        },
        "description": "Semantic grouping chunking with threshold ranges - no size constraints"
    },
    {
        "name": "semantic_grouping_balanced",
        "method_choice": "1",
        "params": {
            "initial_threshold": 0.9,
            "decay_factor": 0.75,
            "min_threshold": 0.3,
            "min_sentences_per_chunk": 2,
            "target_chunk_size": 150,      # Target character count per chunk
            "size_tolerance": 0.5,         # ±50% tolerance for chunk size
            "enable_balanced_chunking": True,  # Enable the new balanced algorithm
        },
        "description": "Balanced semantic grouping - balances semantic coherence with chunk size consistency"
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
        print("Input file not found – aborting.")
        return

    out_dir = Path(input("Enter output directory [training_datasets]: ").strip() or "training_datasets")
    out_dir.mkdir(exist_ok=True)

    # ----- Model preset -----
    current_embedding_model = COMMON_DEFAULTS["embedding_model"]
    print(f"\nCurrent embedding model: {current_embedding_model}")
    if input("Change model? (y/N): ").strip().lower() == "y":
        print("\nQUICK MODEL PRESETS:")
        for key, (model_name, desc) in _MODEL_PRESETS.items():
            print(f"  {key}. {model_name} - {desc}")
        choice = input("Select preset (1-5) or press ENTER to keep current: ").strip()
        if choice in _MODEL_PRESETS:
            selected = _MODEL_PRESETS[choice][0]
            if selected == "auto":
                selected = _recommend_auto_model()
            current_embedding_model = selected
            print(f"Embedding model set to: {selected}")

    # ----- Config selection -----
    print("\nAvailable configurations:")
    for i, cfg in enumerate(RUN_CONFIGURATIONS, 1):
        desc = cfg.get('description', 'No description')
        params_info = []
        
        # Extract key parameters for display
        params = cfg.get('params', {})
        if 'min_sentences_per_chunk' in params:
            params_info.append(f"min_sentences={params['min_sentences_per_chunk']}")
        if 'include_oie' in params and params['include_oie']:
            params_info.append("OIE=Yes")
        if 'semantic_threshold' in params:
            params_info.append(f"threshold={params['semantic_threshold']}")
        if 'initial_threshold' in params and params['initial_threshold'] != 'auto':
            params_info.append(f"init_th={params['initial_threshold']}")
            
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
        print("No configuration selected – aborting.")
        return

    # ----- Workers per config (batch) -----
    cfg_workers = int(input("Workers per config (batch) [1]: ") or "1")

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
        if 'include_oie' in params and params['include_oie']:
            params_summary.append("OIE")
        param_str = f" ({', '.join(params_summary)})" if params_summary else ""
        print(f"  - {cfg['name']}{param_str}")
    print(f"Workers      : per-config={cfg_workers}\n")

    # spawn start-method
    mp.set_start_method("spawn", force=True)

    # Chạy tất cả configs được chọn - use user selections, not COMMON_DEFAULTS
    for i, cfg in enumerate(selected_cfgs, 1):
        print(f"\n[CONFIG {i}/{len(selected_cfgs)}] Starting {cfg['name']}...")
        run_config(
            config=cfg, 
            input_path=input_path, 
            output_dir=out_dir, 
            config_workers=cfg_workers,
            batch_size=BATCH_SIZE,
            embedding_model=current_embedding_model,
            device_preference=COMMON_DEFAULTS["device_preference"],
            enable_text_cleaning=COMMON_DEFAULTS["enable_text_cleaning"]
        )

    print(f"\nAll {len(selected_cfgs)} configuration(s) completed! Output in", out_dir)


# ==== CLI ====================================================================

def main():
    # Display available configurations in help
    config_list = "\n".join([f"    {i+1}. {cfg['name']} - {cfg.get('description', 'No description')}" 
                             for i, cfg in enumerate(RUN_CONFIGURATIONS)])
    
    ap = argparse.ArgumentParser(
        description="Simple multi-config chunking controller",
        epilog=f"Available configurations:\n{config_list}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", required=True, help="Input TSV path")
    ap.add_argument("-o", "--output-dir", default="training_datasets", help="Output directory")
    ap.add_argument("-w", "--workers", type=int, default=1, help="Parallel configs workers")
    ap.add_argument("-c", "--config-workers", type=int, default=1, help="Workers per config (process batches)")
    ap.add_argument("--embedding-model", default=COMMON_DEFAULTS["embedding_model"], help="SentenceTransformer model")
    ap.add_argument("--device", default=COMMON_DEFAULTS["device_preference"], help="cpu | cuda | dml")
    ap.add_argument("--enable-text-cleaning", action="store_true", default=True, 
                    help="Enable text cleaning for better SpaCy sentence segmentation (default: True)")
    ap.add_argument("--disable-text-cleaning", action="store_true", 
                    help="Disable text cleaning, use raw documents (overrides --enable-text-cleaning)")
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
    
    if enable_text_cleaning:
        print("Text cleaning enabled for better SpaCy sentence segmentation")
    else:
        print("Text cleaning disabled - using raw documents")

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
        print("No configurations selected – exiting.")
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
                    enable_text_cleaning=enable_text_cleaning
                ): cfg["name"]
                for cfg in configs
            }
            for fut in as_completed(futs):
                fut.result()
    else:
        for cfg in configs:
            run_config(
                config=cfg, 
                input_path=input_path, 
                output_dir=out_dir, 
                config_workers=args.config_workers,
                batch_size=BATCH_SIZE,
                embedding_model=user_embedding_model,
                device_preference=user_device_preference,
                enable_text_cleaning=enable_text_cleaning
            )

    print("All configurations completed!")


if __name__ == "__main__":
    # Nếu được truyền tham số dòng lệnh thì dùng argparse, ngược lại mở interactive UI.
    import sys

    if len(sys.argv) > 1:
        mp.set_start_method("spawn", force=True)
        main()
    else:
        interactive_ui() 