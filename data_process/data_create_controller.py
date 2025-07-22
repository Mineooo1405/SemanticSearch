import subprocess
import os
import sys

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import json
import pandas as pd
import numpy as np
import gc
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from datetime import datetime
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import re
import atexit
from collections import defaultdict
import traceback
import time
# Import ranking utilities
from Tool.rank_chunks import rank_by_cosine_similarity, rank_by_bm25, rank_by_rrf

# --- Integrated Imports from interactive_chunker.py ---
from Method.Semantic_Grouping import semantic_chunk_passage_from_grouping_logic
from Method.Semantic_Splitter import chunk_passage_text_splitter
from Method.Text_Splitter import chunk_passage_text_splitter

# Attempt to import OIE and Sentence Embedding tools
try:
    from Tool.OIE import extract_relations_from_paragraph
    OIE_AVAILABLE_GLOBALLY = True
    print("[DEBUG] OIE tool imported successfully")
except (ImportError, Exception) as e:
    print(f"[DEBUG] OIE import failed: {e}")
    extract_relations_from_paragraph = None
    OIE_AVAILABLE_GLOBALLY = False

try:
    from Tool.Sentence_Embedding import sentence_embedding  # noqa: F401  # type: ignore
except ImportError:
    # Define a placeholder if it fails
    def sentence_embedding(*args, **kwargs):
        raise ImportError("Sentence_Embedding tool is not available and is required for semantic chunking.")

# Set CSV field size limit
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(int(2**30))

# --- Configuration for the controller ---
PYTHON_EXECUTABLE = sys.executable

# Dynamic batch sizing based on available memory
def _utility_calculate_optimal_batch_size():
    """Calculate optimal batch size based on available RAM"""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb > 16:
            return 1000  # High memory system
        elif available_gb > 8:
            return 500   # Medium memory system
        elif available_gb > 4:
            return 200   # Low memory system
        else:
            return 100   # Very low memory system
    except ImportError:
        print("psutil not available. Using conservative batch size of 200.")
        return 200  # Conservative default

def _utility_recommend_embedding_model():
    """Recommend embedding model based on available GPU/RAM"""
    try:
        # Check GPU memory first
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if gpu_memory_gb >= 6:
                    return "gte-large"
                elif gpu_memory_gb >= 3:
                    return "gte-base" 
                elif gpu_memory_gb >= 2:
                    return "mpnet-base"
                else:
                    return "minilm-l12"
        except ImportError:
            pass
        
        # Fallback to RAM-based recommendation
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb >= 8:
            return "gte-base"
        elif available_gb >= 4:
            return "minilm-l12"
        else:
            return "minilm-l6"
            
    except ImportError:
        return "minilm-l6"  # Most conservative option

def _utility_set_embedding_model(model_key: str) -> bool:
    """Set the embedding model in COMMON_DEFAULTS"""
    if model_key in AVAILABLE_EMBEDDING_MODELS:
        COMMON_DEFAULTS["embedding_model"] = AVAILABLE_EMBEDDING_MODELS[model_key]["model_name"]
        return True
    return False

PROCESSING_BATCH_SIZE = _utility_calculate_optimal_batch_size()  # Dynamic batch size

# Default parameters that will be used for all runs
COMMON_DEFAULTS = {
    "base_output_dir": "./training_datasets",
    "embedding_model": "thenlper/gte-base",  # Default balanced model
    "device_preference": ""  # Empty for auto-detection
}

# Available embedding models with memory requirements
AVAILABLE_EMBEDDING_MODELS = {
    "gte-large": {
        "model_name": "thenlper/gte-large",
        "description": "High quality, large model (1.3B params)",
        "vram_gb": 4.5,
        "dimensions": 1024,
        "recommended_for": "High accuracy tasks, sufficient VRAM"
    },
    "gte-base": {
        "model_name": "thenlper/gte-base", 
        "description": "Good balance of quality and size (220M params)",
        "vram_gb": 2.0,
        "dimensions": 768,
        "recommended_for": "Balanced performance and memory usage"
    },
    "gte-small": {
        "model_name": "thenlper/gte-small",
        "description": "Lightweight model (33M params)",
        "vram_gb": 0.8,
        "dimensions": 384,
        "recommended_for": "Low memory systems"
    },
    "minilm-l6": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Very fast and lightweight (22M params)",
        "vram_gb": 0.5,
        "dimensions": 384,
        "recommended_for": "CPU processing, very low memory"
    },
    "minilm-l12": {
        "model_name": "sentence-transformers/all-MiniLM-L12-v2", 
        "description": "Better quality than L6, still lightweight (33M params)",
        "vram_gb": 0.7,
        "dimensions": 384,
        "recommended_for": "Good compromise for medium systems"
    },
    "mpnet-base": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "description": "High quality general purpose model (110M params)",
        "vram_gb": 1.8,
        "dimensions": 768,
        "recommended_for": "High quality with moderate memory usage"
    }
}

ADAPTIVE_CHUNKING_CONFIG = {
    "target_tokens": 120,           # Target chunk size
    "tolerance": 0.25,              # 25% tolerance (90-150 tokens)
    "min_tokens": 90,               # Minimum acceptable tokens (derived from target and tolerance)
    "max_tokens": 150,              # Maximum acceptable tokens (derived from target and tolerance)
    "strict_max": 200,              # Absolute maximum before forced split
    "merge_threshold": 80,          # Merge chunks smaller than this
    "enable_adaptive": False,       # Disable adaptive validation/merging to keep original chunks
    "preserve_sentences": True,     # Try to preserve sentence boundaries
    "max_iterations": 3             # Maximum adaptive iterations
}

SEMANTIC_GROUPING_DEFAULTS = {
    "initial_threshold": 0.75,
    "decay_factor": 0.85,
    "min_threshold": 0.50,
    "initial_percentile": "80",
    "min_percentile": "20",
    "embedding_batch_size": 32,  
    "min_chunk_len_tokens": ADAPTIVE_CHUNKING_CONFIG["min_tokens"],
    "max_chunk_len_tokens": ADAPTIVE_CHUNKING_CONFIG["max_tokens"],
    "oie_max_triples_per_chunk": 5,
    "oie_token_budget": 40
}

SEMANTIC_SPLITTER_DEFAULTS = {
    "initial_threshold": 0.8,
    "decay_factor": 0.90,
    "min_threshold": 0.30,
    "min_chunk_len": 2,
    "max_chunk_len": 4,
    "window_size": 3,
    "embedding_batch_size": 32,  
    "enable_adaptive": ADAPTIVE_CHUNKING_CONFIG["enable_adaptive"],
    "target_tokens": ADAPTIVE_CHUNKING_CONFIG["target_tokens"],
    "tolerance": ADAPTIVE_CHUNKING_CONFIG["tolerance"],
    "oie_token_budget": 40
}

TEXT_SPLITTER_DEFAULTS = {
    "chunk_size": 500,
    "chunk_overlap": 0,
    "enable_adaptive": ADAPTIVE_CHUNKING_CONFIG["enable_adaptive"],
    "target_tokens": ADAPTIVE_CHUNKING_CONFIG["target_tokens"],
    "tolerance": ADAPTIVE_CHUNKING_CONFIG["tolerance"],
    "oie_token_budget": 40
}

# Define the sequence of runs - 6 configurations for 6 datasets
RUN_CONFIGURATIONS = [
    {
        "name": "text_splitter_without_OIE",
        "method_choice": "3", 
        "include_oie": False, 
        "params": TEXT_SPLITTER_DEFAULTS,
        "description": "Adaptive rule-based chunking optimized for 90-150 tokens (without OIE)"
    },    
    {
        "name": "text_splitter_with_OIE",
        "method_choice": "3", 
        "include_oie": True,
        "save_raw_oie": True,
        "params": TEXT_SPLITTER_DEFAULTS,
        "description": "Adaptive rule-based chunking with OIE enhancement (90-150 tokens)"
    },
    {
        "name": "semantic_grouping_without_OIE",
        "method_choice": "1", 
        "include_oie": False, 
        "params": SEMANTIC_GROUPING_DEFAULTS,
        "description": "Adaptive semantic clustering without order constraint (90-150 tokens, without OIE)"
    },    
    {
        "name": "semantic_grouping_with_OIE",
        "method_choice": "1", 
        "include_oie": True,
        "save_raw_oie": True,
        "params": SEMANTIC_GROUPING_DEFAULTS,
        "description": "Adaptive semantic clustering with OIE enhancement (90-150 tokens)"
    },
    {
        "name": "semantic_splitter_without_OIE",
        "method_choice": "2", 
        "include_oie": False, 
        "params": SEMANTIC_SPLITTER_DEFAULTS,
        "description": "Adaptive sequential semantic chunking with order preservation (90-150 tokens, without OIE)"
    },    
    {
        "name": "semantic_splitter_with_OIE",
        "method_choice": "2", 
        "include_oie": True,
        "save_raw_oie": True,
        "params": SEMANTIC_SPLITTER_DEFAULTS,
        "description": "Adaptive sequential semantic chunking with OIE enhancement (90-150 tokens)"
    }
]

class DatasetController:

    def __init__(self, input_tsv_path: str, output_base_dir: str = "training_datasets", *, auto_start_oie: bool = True):
        self.input_tsv_path = input_tsv_path
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self._auto_start_oie = auto_start_oie  # lưu lại tuỳ chọn
        
        # Memory safety check
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 2:
            print(f"WARNING: Only {available_gb:.1f}GB RAM available. Consider freeing memory before processing large datasets.")
        
        # GPU memory check (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_free = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"GPU: {gpu_memory:.1f}GB total, {gpu_free:.1f}GB reserved")
                if gpu_memory < 4:
                    print("WARNING: Low GPU memory. Consider using CPU for embeddings or reducing batch sizes.")
        except ImportError:
            pass
        
        # Setup logging - FIX Unicode issues
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_base_dir / f"adaptive_controller_run_{self.timestamp}.log"
        
        # Configure logging for Windows compatibility
        import sys
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler (supports UTF-8)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        
        # Console handler (ASCII only to avoid emoji issues)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        # Set to INFO level to show processing progress
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Target models that work best with 90-150 tokens (adaptive range)
        self.target_models = ["knrm", "drmm", "anmm", "dssm", "cdssm", "conv_knrm"]
        
        self._log_safe('info', f"Adaptive Controller initialized for dataset: {input_tsv_path}")
        self._log_safe('info', f"Target models: {', '.join(self.target_models)}")
        self._log_safe('info', f"Adaptive chunk range: {ADAPTIVE_CHUNKING_CONFIG['min_tokens']}-{ADAPTIVE_CHUNKING_CONFIG['max_tokens']} tokens")
        self._log_safe('info', f"Target: {ADAPTIVE_CHUNKING_CONFIG['target_tokens']} tokens ±{ADAPTIVE_CHUNKING_CONFIG['tolerance']*100:.0f}%")
        
        # Log current embedding model
        current_model = COMMON_DEFAULTS["embedding_model"]
        model_info = None
        for key, info in AVAILABLE_EMBEDDING_MODELS.items():
            if info["model_name"] == current_model:
                model_info = info
                break
        
        if model_info:
            self._log_safe('info', f"Embedding Model: {current_model}")
            self._log_safe('info', f"VRAM Required: ~{model_info['vram_gb']}GB | Dimensions: {model_info['dimensions']}")
            # Also print to console for immediate visibility
            print(f"[PROCESS] Using embedding model: {current_model}")
        else:
            self._log_safe('info', f"Embedding Model: {current_model} (custom model)")
            print(f"[PROCESS] Using embedding model: {current_model} (custom model)")
        
        # Validate OIE availability
        self.oie_available = OIE_AVAILABLE_GLOBALLY
        if not self.oie_available:
            self._log_safe('warning', "OIE tool is not available. Configurations set to use OIE will run without it.")
        else:
            self._log_safe('info', "OIE tool appears to be available.")

        # Thử khởi động server nếu chưa chạy (tuỳ theo auto_start_oie)
        if self.oie_available and self._auto_start_oie:
            try:
                from Tool.OIE import _is_port_open
                if not _is_port_open(9000):
                    self._log_safe('warning', "Đang khởi chạy server OpenIE5 (cổng 9000)...")
                    # Gọi OIE để kích hoạt cơ chế auto-start
                    if extract_relations_from_paragraph is not None:  # type: ignore
                        extract_relations_from_paragraph("Ping", port=9000, silent=True)  # type: ignore[arg-type]
                if _is_port_open(9000):
                    self._log_safe('warning', "Đã khởi chạy server OpenIE5 (cổng 9000).")
            except Exception:
                pass

    def _log_safe(self, level: str, message: str):
        """Helper to log messages without emoji processing"""
        getattr(self.logger, level.lower(), self.logger.info)(message)
    
    def _utility_memory_status(self) -> dict:
        """Get current memory status for monitoring"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            status = {
                "total_ram_gb": memory.total / (1024**3),
                "available_ram_gb": memory.available / (1024**3),
                "ram_usage_percent": memory.percent,
                "process_memory_mb": process.memory_info().rss / (1024**2)
            }
            
            # GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    status["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    status["gpu_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
                    status["gpu_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
            except ImportError:
                pass
                
            return status
        except ImportError:
            return {"error": "psutil not available"}
    
    def _utility_check_memory_safety(self, warn_threshold: float = 85.0) -> bool:
        """Check if memory usage is safe to continue processing"""
        status = self._utility_memory_status()
        if "error" in status:
            return True  # Can't check, assume safe
            
        ram_usage = status.get("ram_usage_percent", 0)
        if ram_usage > warn_threshold:
            self._log_safe('warning', f"High RAM usage: {ram_usage:.1f}%. Consider reducing batch size or freeing memory.")
            if ram_usage > 95:
                self._log_safe('error', f"Critical RAM usage: {ram_usage:.1f}%. Processing may fail!")
                return False
        
        return True
    
    def _utility_display_embedding_models(self) -> None:
        """Display available embedding models with recommendations"""
        
        recommended = _utility_recommend_embedding_model()
        current_model = COMMON_DEFAULTS["embedding_model"]
        
        for key, info in AVAILABLE_EMBEDDING_MODELS.items():
            status_flags = []
            if info["model_name"] == current_model:
                status_flags.append("CURRENT")
            if key == recommended:
                status_flags.append("RECOMMENDED")
            
            status_str = f" [{', '.join(status_flags)}]" if status_flags else ""
            
            print(f"\n{key.upper()}{status_str}")
            print(f"   Model: {info['model_name']}")
            print(f"   Description: {info['description']}")
            print(f"   VRAM Required: ~{info['vram_gb']}GB")
            print(f"   Dimensions: {info['dimensions']}")
            print(f"   Best For: {info['recommended_for']}")
    
    def _utility_change_embedding_model(self, model_key: str = None) -> bool:
        """Change the embedding model with validation"""
        if model_key is None:
            self._utility_display_embedding_models()
            return True
            
        if model_key not in AVAILABLE_EMBEDDING_MODELS:
            available_keys = list(AVAILABLE_EMBEDDING_MODELS.keys())
            print(f"Invalid model key '{model_key}'. Available: {available_keys}")
            return False
        
        # Check memory requirements
        model_info = AVAILABLE_EMBEDDING_MODELS[model_key]
        required_vram = model_info["vram_gb"]
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if required_vram > gpu_memory_gb * 0.8:  # Leave 20% buffer
                    print(f"WARNING: Model requires ~{required_vram}GB VRAM but only {gpu_memory_gb:.1f}GB available!")
                    print("Consider using CPU processing or a smaller model.")
                    
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        return False
        except ImportError:
            print("GPU not available, will use CPU processing.")
        
        # Update the model
        old_model = COMMON_DEFAULTS["embedding_model"]
        COMMON_DEFAULTS["embedding_model"] = model_info["model_name"]
        
        print(f"Embedding model changed:")
        print(f"   From: {old_model}")
        print(f"   To: {model_info['model_name']}")
        print(f"   VRAM: ~{required_vram}GB | Dimensions: {model_info['dimensions']}")
        
        return True
    
    def _utility_quick_model_switch(self) -> None:
        """Quick model switching with common presets"""
        print("\nQUICK MODEL PRESETS:")
        print("1.High Quality (gte-large) - Best accuracy, needs 4.5GB+ VRAM")
        print("2.Balanced (gte-base) - Good quality, moderate memory (2GB VRAM)")
        print("3.Fast (minilm-l12) - Quick processing, low memory (0.7GB VRAM)")
        print("4.Ultra-Fast (minilm-l6) - Fastest, minimal memory (0.5GB VRAM)")
        print("5.Auto-select based on system")
        
        choice = input("\nSelect preset (1-5) or press ENTER to skip: ").strip()
        
        model_map = {
            "1": "gte-large",
            "2": "gte-base", 
            "3": "minilm-l12",
            "4": "minilm-l6",
            "5": _utility_recommend_embedding_model()
        }
        
        if choice in model_map:
            model_key = model_map[choice]
            if self._utility_change_embedding_model(model_key):
                print("Model preset applied successfully!")
            else:
                print("Failed to apply preset.")
        elif choice:
            print("Invalid choice.")
    
    def helper_runtime_model_change(self, model_key: str = None) -> bool:
        """Runtime method to change embedding model during processing"""
        if model_key is None:
            print("Available models:")
            for key in AVAILABLE_EMBEDDING_MODELS.keys():
                print(f"   - {key}")
            return False
        
        return self._utility_change_embedding_model(model_key)
    
    def _utility_validate_input_file(self) -> bool:
        """Validate that input TSV file exists and has correct format"""
        if not os.path.exists(self.input_tsv_path):
            self._log_safe('error', f"Input file not found: {self.input_tsv_path}")
            return False
            
        try:
            with open(self.input_tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader, None)
                if not header or len(header) < 3:
                    self._log_safe('error', f"Invalid TSV format. Expected at least 3 columns (query, passage, label), got: {header}")
                    return False
            self._log_safe('info', f"Input TSV file validated successfully with header: {header}")
            return True
            
        except Exception as e:
            self._log_safe('error', f"Error validating input file: {e}")
            return False
    
    def process_execute_chunking_logic(self, df: pd.DataFrame, config: dict) -> List[List[Any]]:
        """
        Memory-efficient chunking logic with garbage collection.
        Takes a DataFrame (batch) and a configuration, returns chunked rows.
        
        Memory scaling factors:
        1. Embedding model: Fixed ~2-4GB VRAM (loaded once, persists)
        2. output_rows accumulation: ~10-50MB per 1000 chunks (main scaling issue)
        3. Temporary sentence processing: ~1-10MB per row (cleared after each row)
        4. Dataset size: Memory grows linearly with output chunks, not input rows
        """
        output_rows = []
        method_choice = config["method_choice"]
        method_params = config["params"].copy() # Use a copy

        # Determine if OIE should be used for this run
        use_oie = config.get("include_oie", False) and self.oie_available
        method_params['include_oie'] = use_oie
        method_params['save_raw_oie'] = use_oie and config.get("save_raw_oie", False)

        chunking_function_map = {
            "1": semantic_chunk_passage_from_grouping_logic,
            "2": chunk_passage_text_splitter,
            "3": chunk_passage_text_splitter,
        }
        chunking_function = chunking_function_map.get(method_choice)

        if not chunking_function:
            self._log_safe('warning', f"No valid chunking function for method_choice '{method_choice}'. Skipping.")
            return []

        # Memory monitoring
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Memory optimization: prevent output_rows accumulation for large datasets
        max_buffer_chunks = 1000  # Clear buffer every 1000 chunks to reduce memory

        for row_idx, row_data in df.iterrows():
            try:
                query_text, passage_text, label = row_data.iloc[0], row_data.iloc[1], row_data.iloc[2]
                query_id = f"Q{row_idx}"
                passage_id = f"P{row_idx}"

                # Log progress every 10 rows to track processing
                if row_idx % 10 == 0:
                    self._log_safe('info', f"[{config['name']}] Processing query_id: {query_id} (row {row_idx})")

                if not isinstance(passage_text, str) or not passage_text.strip():
                    self.logger.warning(f"Skipping row {row_idx} due to empty passage text.")
                    continue

                # Memory check every 100 rows
                if row_idx % 100 == 0:
                    current_memory = process.memory_info().rss / (1024**2)  # MB
                    memory_growth = current_memory - initial_memory
                    if memory_growth > 1000:  # If memory grew by more than 1GB
                        self._log_safe('warning', f"High memory usage detected: {memory_growth:.1f}MB growth. Running GC...")
                        gc.collect()  # Force garbage collection

                # Prepare parameters for the chunking function
                common_params = {
                    'passage_text': passage_text,
                    'doc_id': f"{query_id}_{passage_id}",
                    'output_dir': str(self.output_base_dir / config["name"]),
                    'silent': True,
                    **method_params
                }
                if method_choice in ["1", "2"]: # Semantic methods
                    common_params['embedding_model'] = COMMON_DEFAULTS["embedding_model"]
                    common_params['device'] = COMMON_DEFAULTS["device_preference"]

                # Call the selected chunking function
                chunked_tuples: List[Tuple[str, str, Optional[str]]] = chunking_function(**common_params)

                if not chunked_tuples:
                    self.logger.warning(f"No chunks generated for row {row_idx}. Original passage: {passage_text[:100]}...")
                    continue
                
                # Format output rows
                for gen_chunk_id, chunk_text, oie_string in chunked_tuples:
                    output_rows.append([
                        query_id, query_text, passage_id, passage_text, label,
                        gen_chunk_id, chunk_text, oie_string if oie_string else ""
                    ])
                    
                # Memory optimization: Clear buffer periodically to prevent accumulation
                if len(output_rows) >= max_buffer_chunks:
                    self._log_safe('debug', f"Buffer size reached {len(output_rows)} chunks, clearing periodically...")
                    # Note: In production, would flush to temporary file here
                    # For now, trigger garbage collection more aggressively
                    gc.collect()
                    
                # Clear chunked_tuples to free memory immediately
                del chunked_tuples
                
            except Exception as e:
                self.logger.error(f"Error processing row {row_idx} in config '{config['name']}': {e}", exc_info=False)
        
        # Final garbage collection
        gc.collect()
        final_memory = process.memory_info().rss / (1024**2)  # MB
        self._log_safe('info', f"Batch processed. Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB ({len(output_rows)} chunks)")
        
        return output_rows

    def _run_single_config(self, config: dict) -> Tuple[bool, str, Optional[float]]:
        """Runs a single data creation configuration, processing in batches."""
        task_name = config["name"]
        self._log_safe('info', f"Starting task: {task_name} | {config['description']}")
        
        config_output_dir = self.output_base_dir / task_name
        config_output_dir.mkdir(exist_ok=True)
        final_output_file = config_output_dir / f"output_chunks_final_{self.timestamp}.tsv"
        
        self._log_safe('info', f"Processing in batches of {PROCESSING_BATCH_SIZE}. Final output: {final_output_file}")
        
        is_first_batch = True
        batch_success = True
        total_chunks = 0

        try:
            with pd.read_csv(self.input_tsv_path, sep='\t', engine='python', chunksize=PROCESSING_BATCH_SIZE, header=0, on_bad_lines='warn') as reader:
                for i, batch_df in enumerate(reader, 1):
                    batch_start_row = (i-1) * PROCESSING_BATCH_SIZE
                    batch_end_row = batch_start_row + len(batch_df) - 1
                    self._log_safe('info', f"--- Processing Batch {i} for {task_name} (rows {batch_start_row}-{batch_end_row}) ---")
                    
                    # Clear any cached data before processing
                    gc.collect()
                    
                    chunked_rows = self.process_execute_chunking_logic(batch_df, config)
                    batch_chunks_count = len(chunked_rows)
                    total_chunks += batch_chunks_count

                    # Write to file immediately and clear from memory
                    write_mode = 'w' if is_first_batch else 'a'
                    with open(final_output_file, write_mode, encoding='utf-8', newline='') as f:
                        writer = csv.writer(f, delimiter='\t')
                        if is_first_batch:
                            writer.writerow(['query_id', 'query_text', 'passage_id', 'original_passage', 'label', 'chunk_id', 'chunk_text', 'raw_oie_data'])
                            is_first_batch = False
                        if chunked_rows:
                            writer.writerows(chunked_rows)
                    
                    # Clear batch data from memory immediately
                    del chunked_rows, batch_df
                    gc.collect()
                    
                    self._log_safe('info', f"Batch {i} completed. Generated {batch_chunks_count} chunks. Total chunks so far: {total_chunks}")
                    
                    # Memory usage warning
                    import psutil
                    memory_usage = psutil.virtual_memory().percent
                    if memory_usage > 85:
                        self._log_safe('warning', f"High RAM usage: {memory_usage:.1f}%. Consider reducing batch size.")
            
            self._log_safe('info', f"Total chunks generated for {task_name}: {total_chunks}")

        except Exception as e:
            self._log_safe('error', f"A critical error occurred during batch processing for {task_name}: {e}")
            batch_success = False

        if not batch_success or not os.path.exists(final_output_file):
            return False, "", None

        # Bỏ các bước phân tích / thống kê để giảm log và tăng tốc
        final_compliance = None

        # --- NEW: Rank chunks and keep top/bottom quartile ---
        try:
            # Lấy ngưỡng phân vị từ biến môi trường nếu có
            try:
                up_perc = int(os.getenv("UPPER_PERCENTILE", "80"))
                low_perc = int(os.getenv("LOWER_PERCENTILE", "20"))
            except ValueError:
                up_perc, low_perc = 80, 20

            # Memory management before ranking
            gc.collect()
            self._log_safe('info', f"Starting ranking phase for {task_name}")
            
            ranked_filtered_path = self.helper_rank_and_filter_chunks(str(final_output_file), config_output_dir,
                                                               upper_percentile=up_perc, lower_percentile=low_perc)
            if ranked_filtered_path:
                self._log_safe('warning', f"Ranked & filtered results saved to: {ranked_filtered_path}")
            
            # Force garbage collection after ranking
            gc.collect()
            
        except Exception as e:
            self._log_safe('error', f"Ranking/filtering failed for {task_name}: {e}")
            # Continue processing even if ranking fails
            import traceback
            self._log_safe('debug', f"Ranking error details: {traceback.format_exc()}")
        
        return True, str(final_output_file), final_compliance
    
    def helper_rank_and_filter_chunks(self, chunks_tsv: str, output_dir: Path,
                                upper_percentile: int = 80, lower_percentile: int = 20) -> str:
        """Rank chunks và giữ lại theo ngưỡng phân vị RRF - OPTIMIZED VERSION.

        Args:
            chunks_tsv:     Đường dẫn file TSV chứa các chunk đã sinh.
            output_dir:     Thư mục lưu file đã lọc.
            upper_percentile:Phần trăm trên để lấy POS (mặc định 80).
            lower_percentile:Phần trăm dưới để lấy NEG (mặc định 20).

        Returns:
            Đường dẫn file TSV đã lọc (rỗng nếu lỗi).
        """
        try:
            # Import optimized ranking function
            from Tool.rank_chunks_optimized import rank_and_filter_chunks_optimized
            
            self._log_safe('info', f"Starting optimized ranking for {chunks_tsv}")
            result_path = rank_and_filter_chunks_optimized(
                chunks_tsv=chunks_tsv,
                output_dir=output_dir, 
                upper_percentile=upper_percentile,
                lower_percentile=lower_percentile,
                model_name=COMMON_DEFAULTS["embedding_model"]
            )
            
            if result_path:
                self._log_safe('info', f"Optimized ranking completed: {result_path}")
            else:
                self._log_safe('warning', f"Optimized ranking failed for {chunks_tsv}")
            
            return result_path
            
        except ImportError:
            # Fallback to original implementation
            self._log_safe('warning', "Optimized ranking not available, using fallback...")
            return self._helper_rank_and_filter_chunks_fallback(chunks_tsv, output_dir, upper_percentile, lower_percentile)
        except Exception as e:
            self._log_safe('error', f"Optimized ranking failed: {e}")
            return ""
    
    def _helper_rank_and_filter_chunks_fallback(self, chunks_tsv: str, output_dir: Path,
                                upper_percentile: int = 80, lower_percentile: int = 20) -> str:
        """Fallback to original ranking implementation"""
        import pandas as pd
        import numpy as np
        from pathlib import Path as _Path

        if not os.path.exists(chunks_tsv):
            return ""

        df = pd.read_csv(chunks_tsv, sep='\t', on_bad_lines='warn', engine='python')
        df.columns = df.columns.str.strip()

        required_cols = {'query_id', 'query_text', 'chunk_text', 'chunk_id'}
        if not required_cols.issubset(set(df.columns)):
            return ""

        kept_rows = []

        for query_id, group_df in df.groupby('query_id'):
            try:
                query_text = str(group_df['query_text'].iloc[0])

                cosine_ranked = rank_by_cosine_similarity(query_text, group_df, text_column='chunk_text', model_name=COMMON_DEFAULTS["embedding_model"])
                bm25_ranked   = rank_by_bm25(query_text, group_df, text_column='chunk_text')
                rrf_ranked    = rank_by_rrf(cosine_ranked, bm25_ranked, k=60)

                if rrf_ranked.empty:
                    continue

                scores = rrf_ranked['rrf_score'].to_numpy(dtype=float)
                pos_thr = np.percentile(scores, upper_percentile)
                neg_thr = np.percentile(scores, lower_percentile)

                selected = rrf_ranked[(rrf_ranked['rrf_score'] >= pos_thr) | (rrf_ranked['rrf_score'] <= neg_thr)]
                kept_rows.append(selected)
            except Exception:
                # Skip problematic group silently to avoid interrupting entire process
                continue

        if not kept_rows:
            return ""

        final_df = pd.concat(kept_rows).reset_index(drop=True)
        save_path = output_dir / f"{_Path(chunks_tsv).stem}_rrf_filtered.tsv"
        final_df.to_csv(save_path, sep='\t', index=False)

        # Tạo file chỉ gồm 3 cột query, passage, label
        try:
            simple_df = final_df[['query_text', 'chunk_text', 'label']].copy()
            simple_df.columns = ['query', 'passage', 'label']
            simple_path = output_dir / f"{_Path(chunks_tsv).stem}_rrf_filtered_3col.tsv"
            simple_df.to_csv(simple_path, sep='\t', index=False)
        except Exception:
            simple_path = ""

        return str(save_path)
    
    def helper_extract_output_path(self, stdout: str, task_name: str, config_output_dir: Path) -> str:
        """Extract output file path from stdout"""
        try:
            # Look for output file patterns in stdout
            lines = stdout.split('\n')
            for line in lines:
                if "Output file:" in line and ".tsv" in line:
                    # Extract path from quotes
                    if "'" in line:
                        start = line.find("'") + 1
                        end = line.rfind("'")
                        if start > 0 and end > start:
                            return line[start:end]
            
            # Fallback: look for most recent TSV file in subdirectories
            if config_output_dir.exists():
                tsv_files = list(config_output_dir.glob("**/*.tsv"))
                if tsv_files:
                    most_recent = max(tsv_files, key=lambda f: f.stat().st_mtime)
                    return str(most_recent)
            
            return str(config_output_dir / "output_chunks.tsv")
            
        except Exception as e:
            self.logger.warning(f"Error extracting output path for {task_name}: {e}")
            return str(config_output_dir / "output_chunks.tsv")

    def helper_analyze_chunk_distribution(self, output_file: str, task_name: str) -> None:
        """Analyze token distribution of generated chunks"""
        try:
            if not os.path.exists(output_file):
                self.logger.warning(f"Output file not found for analysis: {output_file}")
                return
            
            df = pd.read_csv(output_file, sep='\t', on_bad_lines='warn', engine='python')
            if 'chunk_text' not in df.columns:
                self.logger.warning(f"Column 'chunk_text' not found in {output_file}. Cannot analyze.")
                return
            df.dropna(subset=['chunk_text'], inplace=True)
            chunk_token_counts = df['chunk_text'].apply(self._utility_count_tokens).tolist()
        
            if not chunk_token_counts:
                self.logger.warning(f"No valid chunks found in {output_file} for analysis.")
                return
            
            # Calculate statistics
            min_tokens = min(chunk_token_counts)
            max_tokens = max(chunk_token_counts)
            avg_tokens = sum(chunk_token_counts) / len(chunk_token_counts)
            
            # Count chunks in target range
            target_min = ADAPTIVE_CHUNKING_CONFIG["min_tokens"]
            target_max = ADAPTIVE_CHUNKING_CONFIG["max_tokens"]
            in_range = sum(1 for count in chunk_token_counts if target_min <= count <= target_max)
            in_range_percent = (in_range / len(chunk_token_counts)) * 100 if chunk_token_counts else 0
            
            self.logger.info(f"Chunk Analysis for {task_name}:")
            self.logger.info(f"   Total chunks: {len(chunk_token_counts)}")
            self.logger.info(f"   Token range: {min_tokens}-{max_tokens} (avg: {avg_tokens:.1f})")
            self.logger.info(f"   In target range ({target_min}-{target_max}): {in_range}/{len(chunk_token_counts)} ({in_range_percent:.1f}%)")
                
        except Exception as e:
            self.logger.warning(f"Error analyzing chunk distribution for {task_name}: {e}")

    def _utility_count_tokens(self, text: str) -> int:
        """Simple token counting for analysis"""
        if not isinstance(text, str):
            return 0
        return len(re.findall(r'\b\w+\b|[^\w\s]', text))

    def helper_create_summary(self, results: List[dict], successful_tasks: int) -> Dict[str, Any]:
        """Create summary of all tasks"""
        return {
            "controller_run_timestamp": self.timestamp,
            "input_tsv_file": str(self.input_tsv_path),
            "output_base_directory": str(self.output_base_dir),
            "target_models": self.target_models,
            "adaptive_chunking_config": ADAPTIVE_CHUNKING_CONFIG,
            "total_configurations": len(RUN_CONFIGURATIONS),
            "successful_tasks": successful_tasks,
            "failed_tasks": len(RUN_CONFIGURATIONS) - successful_tasks,
            "task_results": results
        }

    def helper_save_and_print_summary(self, summary: Dict[str, Any]) -> None:
        """Save and print final summary"""
        results_file = self.output_base_dir / f"controller_results_{self.timestamp}.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Detailed results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results file: {e}")
        
        self.helper_print_final_summary(summary)

        # Tắt server OpenIE sau khi hoàn tất
        try:
            from Tool.OIE import terminate_openie_processes
            terminate_openie_processes(port=9000, silent=True)
            self._log_safe('warning', "Đã tắt server OpenIE5 (cổng 9000).")
        except Exception:
            pass

    # Simplified approach: Replace create_all_datasets method
    def data_create_controller_main(self, use_parallel: bool = False, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Create all 6 adaptive datasets with simple re-chunking loop"""
        
        if use_parallel:
            return self.process_create_all_datasets_parallel(max_workers)
        else:
            return self.process_create_all_datasets_sequential()

    def process_create_all_datasets_sequential(self) -> Dict[str, Any]:
        """Sequential dataset creation."""
        self._log_safe('info', "="*80)
        self._log_safe('info', "STARTING SEQUENTIAL ADAPTIVE DATASET CREATION")
        self._log_safe('info', "="*80)
        
        if not self._utility_validate_input_file():
            self._log_safe('error', "Input file validation failed. Aborting.")
            return {"success": False, "results": []}
        
        # Only run without_OIE configurations in normal mode
        configs_to_run = [c for c in RUN_CONFIGURATIONS if not c.get("include_oie", False)]
        
        results = []
        successful_tasks = 0
        
        for i, config in enumerate(configs_to_run, 1):
            self._log_safe('info', f"\n>>> Configuration {i}/{len(configs_to_run)}: {config['name']} <<<")
            start_time = time.time()
            
            success, output_file, final_compliance = self._run_single_config(config)

            end_time = time.time()

            task_result = {
                "config_name": config["name"], "description": config["description"],
                "success": success, "output_file": output_file,
                "execution_time_seconds": end_time - start_time,
                "final_compliance_rate": final_compliance
            }
            results.append(task_result)
            
            if success:
                successful_tasks += 1
                self._log_safe('info', f"Completed {config['name']} in {task_result['execution_time_seconds']:.1f}s")
            else:
                self._log_safe('error', f"Failed {config['name']} after {task_result['execution_time_seconds']:.1f}s")
        
        summary = self.helper_create_summary(results, successful_tasks)
        self.helper_save_and_print_summary(summary)
        return summary

    def process_create_all_datasets_parallel(self, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Parallel dataset creation using ProcessPoolExecutor"""
        self.logger.info("="*80)
        self.logger.info("STARTING PARALLEL ADAPTIVE DATASET CREATION")
        self.logger.info("="*80)
        
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 4)
        self.logger.info(f"Using up to {max_workers} parallel workers")
        
        if not self._utility_validate_input_file():
            self.logger.error("Input file validation failed. Aborting.")
            return {"success": False, "results": []}
        
        # In normal mode, only run non-OIE configurations
        non_oie_configs = [c for c in RUN_CONFIGURATIONS if not c.get("include_oie")]
        
        results = []
        successful_tasks = 0
        
        # Run non-OIE configurations in parallel
        if non_oie_configs:
            self.logger.info(f"Running {len(non_oie_configs)} non-OIE configurations in parallel...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {
                    executor.submit(helper_run_chunking_config_process, config, self.input_tsv_path, str(self.output_base_dir), auto_start_oie=False, embedding_model=COMMON_DEFAULTS["embedding_model"]): config
                    for config in non_oie_configs
                }
                for future in as_completed(future_to_config):
                    config_name = future_to_config[future]['name']
                    try:
                        result = future.result()
                        results.append(result)
                        if result["success"]: successful_tasks += 1
                    except Exception as e:
                        self.logger.error(f"Process for {config_name} generated an exception: {e}")
                        results.append({"config_name": config_name, "success": False, "error": str(e)})

        # Note: OIE configurations are not run in normal mode

        summary = self.helper_create_summary(results, successful_tasks)
        self.helper_save_and_print_summary(summary)
        return summary

    def helper_get_compliance_rate(self, output_file: str) -> Optional[float]:
        """Get compliance rate from output file."""
        try:
            if not os.path.exists(output_file): return None
            
            df = pd.read_csv(output_file, sep='\t', on_bad_lines='warn', engine='python')
            if 'chunk_text' not in df.columns:
                self.logger.warning(f"Column 'chunk_text' not found in {output_file}. Cannot calculate compliance.")
                return None
            df.dropna(subset=['chunk_text'], inplace=True)
            chunk_token_counts = df['chunk_text'].apply(self._utility_count_tokens).tolist()

            if not chunk_token_counts: return 0.0
            
            target_min = ADAPTIVE_CHUNKING_CONFIG["min_tokens"]
            target_max = ADAPTIVE_CHUNKING_CONFIG["max_tokens"]
            in_range = sum(1 for count in chunk_token_counts if target_min <= count <= target_max)
            return (in_range / len(chunk_token_counts)) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating compliance for {output_file}: {e}")
            return None

    def helper_print_final_summary(self, summary: Dict[str, Any]) -> None:
        """Print final summary to console"""
        self.logger.info("\n" + "="*80)
        self.logger.info("ADAPTIVE DATASET CREATION SUMMARY")
        self.logger.info("="*80)
        
        for result in summary["task_results"]:
            status = "SUCCESS" if result.get("success") else "FAILED"
            compliance_str = ""
            if result.get("success"):
                compliance = result.get("final_compliance_rate")
                compliance_str = f"| Compliance: {compliance:.1f}%" if compliance is not None else ""
            self.logger.info(f"{status} - {result.get('config_name', 'Unknown Task')} {compliance_str}")
        
        self.logger.info("-" * 80)
        successful_tasks = summary.get("successful_tasks", 0)
        total_tasks = summary.get("total_configurations", len(summary.get("task_results", [])))
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        self.logger.info(f"Total configurations run: {total_tasks}")
        self.logger.info(f"Success rate: {success_rate:.1f}% ({successful_tasks}/{total_tasks})")
        
        if successful_tasks > 0:
            self.logger.info(f"\nAll outputs saved to: {summary['output_base_directory']}")
            self.logger.info("="*80)

    # ============================================================
    # New helper: Augment existing TSV bằng OIE song song
    # ============================================================
    def process_augment_with_oie(self, source_tsv: str, threads: int = 8, chunk_rows: int = 1000) -> str:
        try:
            if not self.oie_available:
                self._log_safe('error', 'OIE tool không khả dụng.')
                return ""

            import pandas as _pd
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from pathlib import Path as _Path
            from Method.Text_Splitter import format_oie_triples_to_string  # dùng hàm format chung

            if not os.path.exists(source_tsv):
                self._log_safe('error', f'Không tìm thấy file để augment: {source_tsv}')
                return ""

            # Xác định thư mục đích dựa vào cấu hình gốc (sibling _with_OIE)
            src_path_obj = _Path(source_tsv)
            orig_dir = src_path_obj.parent
            if orig_dir.name.endswith("_without_OIE"):
                target_dir = orig_dir.parent / orig_dir.name.replace("_without_OIE", "_with_OIE")
            else:
                target_dir = orig_dir / "with_OIE"
            target_dir.mkdir(exist_ok=True)

            out_path = target_dir / f"{src_path_obj.stem}_with_OIE.tsv"

            reader_iter = _pd.read_csv(source_tsv, sep='\t', on_bad_lines='warn', engine='python', chunksize=chunk_rows)

            first_chunk = True

            def enrich_chunk(chunk_df: _pd.DataFrame):
                # đảm bảo cột raw_oie_data
                if 'raw_oie_data' not in chunk_df.columns:
                    chunk_df['raw_oie_data'] = ''
                if chunk_df['raw_oie_data'].dtype != 'O':
                    chunk_df['raw_oie_data'] = chunk_df['raw_oie_data'].astype(object)

                # chọn hàng cần OIE
                todo_idx_local = chunk_df.index[chunk_df['raw_oie_data'].isna() | (chunk_df['raw_oie_data'] == '')].tolist()
                if not todo_idx_local:
                    return chunk_df  # nothing to do

                self._log_safe('info', f"[augment_with_oie] Chunk size={len(chunk_df)} | need OIE for {len(todo_idx_local)} rows")

                def _process_local(local_idx):
                    try:
                        qid = str(chunk_df.at[local_idx, 'query_id']) if 'query_id' in chunk_df.columns else 'N/A'
                        cid = str(chunk_df.at[local_idx, 'chunk_id']) if 'chunk_id' in chunk_df.columns else f'row{local_idx}'
                        # Log every 10th OIE processing to avoid spam
                        if local_idx % 10 == 0:
                            self._log_safe('info', f"[augment_with_oie] Processing query_id={qid} | chunk_id={cid}")
                    except Exception:
                        pass

                    txt = str(chunk_df.at[local_idx, 'chunk_text'])
                    triples = extract_relations_from_paragraph(txt, silent=True)  # type: ignore[arg-type]
                    return local_idx, (format_oie_triples_to_string(triples) if triples else '')

                with ThreadPoolExecutor(max_workers=threads) as executor:
                    fut_map = {executor.submit(_process_local, li): li for li in todo_idx_local}
                    for fut in as_completed(fut_map):
                        idx_local, oie_str_local = fut.result()
                        chunk_df.at[idx_local, 'raw_oie_data'] = oie_str_local

                return chunk_df

            # Process each chunk and write to output incrementally
            for chunk_df in reader_iter:
                if 'chunk_text' not in chunk_df.columns:
                    self._log_safe('error', f"'chunk_text' column missing in chunk; abort.")
                    return ""

                processed_chunk = enrich_chunk(chunk_df)
                processed_chunk.to_csv(out_path, sep='\t', index=False, mode='w' if first_chunk else 'a', header=first_chunk)
                first_chunk = False

            # Tạo bản 3 cột (đọc lại file mới, chỉ lấy 3 cột cần thiết)
            try:
                simple_df = _pd.read_csv(out_path, sep='\t', usecols=['query_text', 'chunk_text', 'label'], on_bad_lines='skip', engine='python')  # type: ignore[arg-type]
                simple_df.columns = ['query', 'passage', 'label']
                simple_df.to_csv(out_path.with_name(out_path.stem + '_3col.tsv'), sep='\t', index=False)
            except Exception:
                pass

            # --- NEW: áp dụng RRF filter cho file augmented ---
            try:
                try:
                    up_perc = int(os.getenv("UPPER_PERCENTILE", "80"))
                    low_perc = int(os.getenv("LOWER_PERCENTILE", "20"))
                except ValueError:
                    up_perc, low_perc = 80, 20

                filtered_path = self.helper_rank_and_filter_chunks(str(out_path), out_path.parent, upper_percentile=up_perc, lower_percentile=low_perc)
                if filtered_path:
                    self._log_safe('warning', f"RRF filtered file saved to: {filtered_path}")
            except Exception as e:
                self._log_safe('error', f"RRF filtering failed for augmented file {out_path}: {e}")

            self._log_safe('info', f'Đã lưu file augment: {out_path}')
            return str(out_path)
        except Exception as e:
            self._log_safe('error', f'Lỗi augment OIE: {e}')
            return ""

    # ============================================================
    # New high-level: Hai pha – chunk trước, OIE sau
    # ============================================================
    def process_create_all_datasets_two_phase(self, *, non_oie_parallel: bool = True, max_workers: Optional[int] = None, oie_threads: int = 8):
        """Thực hiện 2 giai đoạn: (1) sinh chunks không OIE, (2) augment OIE."""

        # ---------------- PHA 1: without_OIE ----------------
        phase1_controller = self  

        configs_phase1 = [c for c in RUN_CONFIGURATIONS if not c.get('include_oie')]
        if not configs_phase1:
            self._log_safe('error', 'Không tìm thấy cấu hình without_OIE.')
            return

        results_phase1 = []
        succ1 = 0

        if non_oie_parallel:
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count(), 4)
            self._log_safe('info', f'Pha 1: chạy song song (max_workers={max_workers})')

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                fut_to_cfg = {executor.submit(helper_run_chunking_config_process, cfg, phase1_controller.input_tsv_path, str(phase1_controller.output_base_dir), auto_start_oie=False, embedding_model=COMMON_DEFAULTS["embedding_model"]): cfg for cfg in configs_phase1}
                for fut in as_completed(fut_to_cfg):
                    result = fut.result()
                    results_phase1.append(result)
                    if result.get('success'):
                        succ1 += 1
        else:
            self._log_safe('info', 'Pha 1: chạy tuần tự.')
            for cfg in configs_phase1:
                success, out_file, _ = phase1_controller._run_single_config(cfg)
                results_phase1.append({'config_name': cfg['name'], 'success': success, 'output_file': out_file})
                if success:
                    succ1 += 1

        # ---------------- PHA 2: augment OIE ----------------
        try:
            from Tool.OIE import _is_port_open, _start_openie_server
            if not _is_port_open(9000):
                self._log_safe('warning', 'Chưa có server OpenIE5 – đang khởi động…')
                _start_openie_server(9000)
                # Chờ tối đa 30s
                for _ in range(30):
                    if _is_port_open(9000):
                        break
                    time.sleep(1)
        except Exception:
            self._log_safe('error', 'Không thể khởi động server OpenIE5. Bỏ qua pha 2.')
            return

        augmented_files = []
        # ------ Song song cấp FILE: augment OIE dùng ProcessPoolExecutor ------
        succ_file_paths = [res.get('output_file', '') for res in results_phase1 if res.get('success') and res.get('output_file')]

        if succ_file_paths:
            max_aug_workers = min(len(succ_file_paths), max(1, multiprocessing.cpu_count() // 2))
            self._log_safe('info', f'Pha 2: augment {len(succ_file_paths)} file song song (workers={max_aug_workers})')

            with ProcessPoolExecutor(max_workers=max_aug_workers) as executor:
                fut_to_src = {executor.submit(helper_run_augment_process, path, str(self.output_base_dir), oie_threads, COMMON_DEFAULTS["embedding_model"]): path for path in succ_file_paths}
                for fut in as_completed(fut_to_src):
                    try:
                        result_path = fut.result()
                    except Exception as e:
                        self._log_safe('error', f'Augment process for {fut_to_src[fut]} failed: {e}')
                        result_path = ''
                    augmented_files.append(result_path)
        else:
            self._log_safe('warning', 'Không có file thành công ở pha 1 để augment.')

        self._log_safe('warning', f'Đã hoàn tất hai pha. File augment: {augmented_files}')

        # Tắt OpenIE server sau khi hoàn thành pha 2
        try:
            from Tool.OIE import terminate_openie_processes
            terminate_openie_processes(port=9000, silent=True)
            self._log_safe('warning', 'Đã tắt server OpenIE5 (cổng 9000).')
        except Exception:
            self._log_safe('error', 'Không thể tắt server OpenIE5.')

# Process-based worker function for parallel execution
def helper_run_chunking_config_process(config: dict, input_tsv_path: str, output_base_dir: str, *, auto_start_oie: bool = True, embedding_model: str = None) -> dict:
    """Process-based worker function to run a single chunking configuration.

    auto_start_oie:
        Quyết định có tự khởi động server OIE5 trong tiến trình con hay không.
        Đối với giai đoạn without_OIE nên đặt False để tránh khởi động thừa.
    embedding_model:
        Model embedding để sử dụng, ghi đè COMMON_DEFAULTS nếu được cung cấp.
    """
    if embedding_model:
        COMMON_DEFAULTS["embedding_model"] = embedding_model
    
    controller = DatasetController(input_tsv_path, output_base_dir, auto_start_oie=auto_start_oie)
    start_time = time.time()
        
    success, output_file, final_compliance = controller._run_single_config(config)
        
    end_time = time.time()
        
    return {
        "config_name": config["name"], "description": config["description"],
        "success": success, "output_file": output_file or "",
            "execution_time_seconds": end_time - start_time,
            "final_compliance_rate": final_compliance
        }

# -------------------------------------------------------------------
# Worker cho pha 2 – augment OIE song song
# -------------------------------------------------------------------

def helper_run_augment_process(source_tsv: str, output_base_dir: str, oie_threads: int, embedding_model: str = None) -> str:
    """Process-based worker: augment single TSV with OIE and RRF filter."""
    # Update embedding model in subprocess if provided
    if embedding_model:
        COMMON_DEFAULTS["embedding_model"] = embedding_model
    
    controller = DatasetController(source_tsv, output_base_dir, auto_start_oie=False)
    return controller.process_augment_with_oie(source_tsv, threads=oie_threads)

# -------------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------------
def data_create_controller_main():
    """Main function to run the adaptive controller"""
    global RUN_CONFIGURATIONS  # override inside main
    print("Integrated Adaptive Dataset Controller for Neural Ranking Models")
    print("="*60)
    
    # Get input parameters
    input_tsv_path = input("Enter path to input TSV file (query<tab>passage<tab>label): ").strip()
    if not input_tsv_path:
        print("ERROR: Input file path is required!")
        return
    
    output_base_dir = input("Enter output base directory (default: ./training_datasets): ").strip() or "./training_datasets"
    
    # --- NEW: Embedding model selection ---
    print(f"\nCurrent embedding model: {COMMON_DEFAULTS['embedding_model']}")
    change_model = input("Do you want to change the embedding model? (y/n, default: n): ").strip().lower()
    
    if change_model == 'y':
        # Create temporary controller to access utility methods
        temp_controller = DatasetController.__new__(DatasetController)
        
        # Show quick presets first
        temp_controller._utility_quick_model_switch()
        
        # If user wants more detailed selection
        advanced = input("\nWant to see all models with details? (y/n): ").strip().lower()
        if advanced == 'y':
            temp_controller._utility_display_embedding_models()
            model_choice = input("\nEnter model key manually: ").strip().lower()
            if model_choice and model_choice in AVAILABLE_EMBEDDING_MODELS:
                _utility_set_embedding_model(model_choice)
                model_info = AVAILABLE_EMBEDDING_MODELS[model_choice]
                print(f"Selected: {model_info['model_name']}")
    
    # Display final selected model
    final_model = COMMON_DEFAULTS['embedding_model']
    model_info = None
    for info in AVAILABLE_EMBEDDING_MODELS.values():
        if info["model_name"] == final_model:
            model_info = info
            break
    
    if model_info:
        print(f"\nFinal embedding model: {final_model}")
        print(f"   VRAM: ~{model_info['vram_gb']}GB | Dimensions: {model_info['dimensions']}")
    else:
        print(f"\nFinal embedding model: {final_model} (custom)")
    
    mode_choice = input("\nChạy hai pha Without→With OIE? (y/n, default: n): ").strip().lower()
    two_phase = mode_choice == 'y'

    parallel_choice = input("Use parallel processing cho pha chunk? (y/n, default: n): ").strip().lower()
    use_parallel = parallel_choice == 'y'
    
    max_workers = None
    if use_parallel:
        max_workers_input = input(f"Max parallel workers (default: auto, max: {multiprocessing.cpu_count()}): ").strip()
        if max_workers_input.isdigit():
            max_workers = min(int(max_workers_input), multiprocessing.cpu_count())

    # --- NEW: allow user to pick specific configurations ---
    print("\nAvailable configurations:")
    for cfg in RUN_CONFIGURATIONS:
        print(f"  - {cfg['name']}")

    selected_cfgs = input("Enter configuration names to run (comma-separated) or press ENTER for ALL: ").strip()

    if selected_cfgs:
        chosen = {name.strip() for name in selected_cfgs.split(',') if name.strip()}
        RUN_CONFIGURATIONS = [c for c in RUN_CONFIGURATIONS if c['name'] in chosen]
        if not RUN_CONFIGURATIONS:
            print("No matching configuration names found. Exiting.")
            return
    
    # --- NEW: ask dynamic percentiles for RRF filtering ---
    up_perc_input = input("Upper percentile for POS (default 80): ").strip()
    low_perc_input = input("Lower percentile for NEG (default 20): ").strip()

    if up_perc_input.isdigit():
        os.environ["UPPER_PERCENTILE"] = up_perc_input
    if low_perc_input.isdigit():
        os.environ["LOWER_PERCENTILE"] = low_perc_input

    print(f"\nConfiguration confirmed. Using embedding model: {COMMON_DEFAULTS['embedding_model']}")
    
    if two_phase:
        print("Mode: Two-phase processing (Without OIE → With OIE)")
    else:
        print("Mode: Single-phase processing (Without OIE only)")
    
    print("Starting process...")
    
    try:
        # In two-phase mode, don't start OIE server initially - it will be started in phase 2
        controller = DatasetController(input_tsv_path, output_base_dir, auto_start_oie=False)

        if two_phase:
            controller.process_create_all_datasets_two_phase(non_oie_parallel=use_parallel, max_workers=max_workers)
        else:
            controller.data_create_controller_main(use_parallel=use_parallel, max_workers=max_workers)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"A critical error occurred in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    start_time = time.time()
    
    data_create_controller_main()
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")