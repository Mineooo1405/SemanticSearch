"""
Integrated Adaptive Dataset Controller for Neural Ranking Models

This module provides comprehensive data processing capabilities for creating
training datasets optimized for neural ranking models like KNRM, DRMM, etc.
Supports multiple chunking strategies with DirectML GPU acceleration.
"""

# Standard library imports
import subprocess
import os
import sys
import psutil

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# Core data processing imports
import json
import pandas as pd
import pandas as _pd  # For utility functions
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

# Local imports - ranking utilities
from Tool.rank_chunks import rank_by_cosine_similarity, rank_by_bm25, rank_by_rrf

# Local imports - chunking methods
from Method.Semantic_Grouping import semantic_chunk_passage_from_grouping_logic
from Method.Semantic_Splitter import chunk_passage_text_splitter
from Method.Text_Splitter import chunk_passage_text_splitter

# Optional imports with graceful error handling
try:
    from Tool.OIE import extract_relations_from_paragraph
    OIE_AVAILABLE_GLOBALLY = True
except (ImportError, Exception) as e:
    print(f"[DEBUG] OIE import failed: {e}")
    extract_relations_from_paragraph = None
    OIE_AVAILABLE_GLOBALLY = False

import importlib
try:
    from Tool.Sentence_Embedding import sentence_embedding  # noqa: F401  # type: ignore
except ImportError:
    # Placeholder function for missing sentence embedding tool
    def sentence_embedding(*args, **kwargs):
        raise ImportError("Sentence_Embedding tool is not available and is required for semantic chunking.")

# Configure CSV field size limits for handling large text content
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(int(2**30))

# DirectML support for AMD GPU acceleration with error handling
try:
    torch_directml = importlib.import_module("torch_directml")
    if not (hasattr(torch_directml, 'is_available') and callable(torch_directml.is_available) and
            hasattr(torch_directml, 'device') and callable(torch_directml.device)):
        torch_directml = None
except ImportError:
    torch_directml = None
except Exception:
    torch_directml = None

# =============================================================================
# CONFIGURATION CONSTANTS AND UTILITY FUNCTIONS
# =============================================================================

PYTHON_EXECUTABLE = sys.executable

# Dynamic batch sizing based on available system memory
def _utility_calculate_optimal_batch_size():
    """Calculate optimal batch size based on available RAM.
    
    Returns:
        int: Optimal batch size for processing chunks
    """
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
    """Recommend optimal embedding model based on available GPU/RAM resources.
    
    Returns:
        str: Recommended model key from AVAILABLE_EMBEDDING_MODELS
    """
    try:
        # Check GPU memory first for DirectML/CUDA support
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
        
        # Fallback to RAM-based recommendation for CPU processing
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb >= 8:
            return "gte-base"
        elif available_gb >= 4:
            return "minilm-l12"
        else:
            return "minilm-l6"
            
    except ImportError:
        return "minilm-l6"  # Most conservative fallback option

def _utility_calculate_optimal_ranking_workers():
    """Calculate optimal worker count for ranking operations.
    
    Ranking operations are memory-intensive due to embedding computations.
    This function calculates a conservative worker count to prevent OOM errors.
    
    Returns:
        int: Optimal number of workers for ranking phase
    """
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # Each ranking worker needs approximately 1.5GB for embedding operations
        memory_per_ranking_worker = 1.5  # GB
        max_by_memory = max(1, int((available_ram_gb * 0.6) / memory_per_ranking_worker))
        
        # Cap at 4 workers for ranking efficiency (diminishing returns beyond this)
        optimal_ranking_workers = min(multiprocessing.cpu_count() // 2, max_by_memory, 4)
        
        return max(1, optimal_ranking_workers)
    except ImportError:
        return 2  # Conservative default when psutil unavailable

def _utility_calculate_total_vram_requirement(num_workers: int, embedding_model: str = None) -> float:
    """Calculate total VRAM requirement for concurrent worker processes.
    
    Computes memory requirements for running multiple embedding workers simultaneously,
    accounting for model size, buffer requirements, and DirectML/CUDA compatibility.
    
    Args:
        num_workers (int): Number of concurrent worker processes
        embedding_model (str, optional): Target embedding model name. 
                                       Uses default from COMMON_DEFAULTS if None.
        
    Returns:
        float: Total VRAM requirement in GB for all workers combined
        
    Note:
        Each worker loads its own model instance to avoid memory conflicts.
        Includes safety buffer for processing overhead and model initialization.
    """
    if embedding_model is None:
        embedding_model = COMMON_DEFAULTS["embedding_model"]
    
    # Lookup VRAM requirement for specified model
    model_vram = 2.0  # Conservative default for unknown models
    for model_info in AVAILABLE_EMBEDDING_MODELS.values():
        if model_info["model_name"] == embedding_model:
            model_vram = model_info["vram_gb"]
            break
    
    # Calculate total requirement: dedicated model per worker + processing buffer
    buffer_per_worker = 0.5  # GB safety margin for processing overhead
    total_vram = num_workers * (model_vram + buffer_per_worker)
    
    return total_vram

def _utility_check_vram_capacity(num_workers: int, embedding_model: str = None) -> dict:
    """Validate system VRAM capacity for concurrent worker processes.
    
    Analyzes available GPU memory against requirements for multiple workers,
    providing detailed capacity assessment and optimization recommendations.
    
    Args:
        num_workers (int): Number of concurrent worker processes planned
        embedding_model (str, optional): Target embedding model. Uses default if None.
        
    Returns:
        dict: Comprehensive capacity analysis containing:
            - has_gpu (bool): GPU availability status
            - has_capacity (bool): Whether sufficient VRAM exists
            - total_vram_gb (float): Available GPU memory in GB
            - required_vram_gb (float): Memory needed for specified configuration
            - utilization_percent (float): Percentage of GPU memory that would be used
            - recommendation (str): Optimization advice for configuration
    """
    try:
        import torch
        
        # Check for DirectML support first (AMD GPUs)
        try:
            import torch_directml
            if torch_directml.is_available():
                # DirectML doesn't provide detailed memory info
                required_vram_gb = _utility_calculate_total_vram_requirement(num_workers, embedding_model)
                return {
                    "has_gpu": True,
                    "has_capacity": True,  # Assume capacity exists for DirectML
                    "total_vram_gb": "Unknown (DirectML)",
                    "required_vram_gb": required_vram_gb,
                    "utilization_percent": "Unknown",
                    "recommendation": f"DirectML detected. Estimated requirement: {required_vram_gb:.1f}GB"
                }
        except ImportError:
            pass
        
        # Check CUDA GPUs
        if not torch.cuda.is_available():
            return {
                "has_gpu": False,
                "recommendation": "No GPU available. Use CPU processing or install DirectML/CUDA PyTorch",
                "total_vram_gb": 0,
                "required_vram_gb": 0
            }
        
        # Analyze CUDA GPU capacity
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        required_vram_gb = _utility_calculate_total_vram_requirement(num_workers, embedding_model)
        
        has_capacity = total_vram_gb >= required_vram_gb
        utilization_percent = (required_vram_gb / total_vram_gb) * 100
        
        return {
            "has_gpu": True,
            "has_capacity": has_capacity,
            "total_vram_gb": total_vram_gb,
            "required_vram_gb": required_vram_gb,
            "utilization_percent": utilization_percent,
            "recommendation": _get_vram_recommendation(has_capacity, utilization_percent, num_workers)
        }
    except ImportError:
        return {
            "has_gpu": False,
            "recommendation": "PyTorch not available - install PyTorch with GPU support",
            "total_vram_gb": 0,
            "required_vram_gb": 0
        }

def _get_vram_recommendation(has_capacity: bool, utilization_percent: float, num_workers: int) -> str:
    """Generate optimization recommendations based on VRAM utilization analysis.
    
    Provides specific guidance for GPU memory usage optimization based on
    current capacity assessment and utilization metrics.
    
    Args:
        has_capacity (bool): Whether system has sufficient VRAM
        utilization_percent (float): Percentage of VRAM that would be utilized
        num_workers (int): Current number of workers planned
        
    Returns:
        str: Specific recommendation for optimizing GPU memory usage
    """
    if not has_capacity:
        suggested_workers = max(1, num_workers // 2)
        return f"Insufficient VRAM! Reduce workers to {suggested_workers} or use smaller embedding model"
    elif utilization_percent > 85:
        return f"High VRAM usage ({utilization_percent:.1f}%). Consider reducing workers or batch size for stability"
    elif utilization_percent > 70:
        return f"Moderate VRAM usage ({utilization_percent:.1f}%). Good balance for performance"
    else:
        return f"Low VRAM usage ({utilization_percent:.1f}%). Can potentially increase workers for better throughput"

def _utility_monitor_gpu_usage():
    """Monitor GPU memory usage with DirectML and CUDA support.
    
    Attempts to detect and report GPU information for both DirectML (AMD) 
    and CUDA (NVIDIA) devices. Provides detailed memory usage for optimization.
    
    Returns:
        dict: GPU usage information including type, device count, and memory stats
    """
    try:
        import torch
        
        # Try DirectML first for AMD GPU support
        try:
            import torch_directml
            if torch_directml.is_available():
                device_count = torch_directml.device_count()
                return {
                    "type": "DirectML",
                    "device_count": device_count,
                    "device_name": torch_directml.device_name(0) if device_count > 0 else "Unknown DirectML Device",
                    "available": True,
                    "note": "DirectML memory monitoring limited"
                }
        except ImportError:
            pass
        
        # Check CUDA GPUs as fallback
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            usage_info = {"type": "CUDA"}
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024**3)   # GB
                total = props.total_memory / (1024**3)                  # GB
                
                usage_info[f"gpu_{i}"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved, 
                    "total_gb": total,
                    "utilization_percent": (allocated / total) * 100,
                    "device_name": props.name
                }
            
            return usage_info
        else:
            return {"type": "CPU", "error": "No GPUs available - using CPU"}
    except ImportError:
        return {"error": "PyTorch not available"}

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
    "device_preference": "dml"  # Empty for auto-detection
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
    "oie_max_triples_per_chunk": 5,
    "oie_token_budget": 40
}

SEMANTIC_SPLITTER_DEFAULTS = {
    "initial_threshold": 0.8,
    "decay_factor": 0.90,
    "min_threshold": 0.30,
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
    """Advanced dataset controller for adaptive semantic chunking and training data generation.
    
    Manages the complete pipeline for creating optimized training datasets from raw input data,
    supporting multiple chunking strategies with DirectML/CUDA acceleration and adaptive
    parameter tuning for various information retrieval models.
    
    Key Features:
        - Multi-strategy chunking (semantic grouping, semantic splitting, text splitting)
        - DirectML and CUDA GPU acceleration support
        - Adaptive parameter optimization based on data characteristics
        - Memory-efficient processing with configurable batch sizes
        - OpenIE-based relationship extraction integration
        - Comprehensive logging and progress tracking
    
    Attributes:
        input_tsv_path (str): Path to input TSV file containing raw dataset
        output_base_dir (Path): Base directory for generated training datasets
        target_models (list): Supported IR models for optimization
        logger (Logger): Configured logging instance for tracking operations
    """

    def __init__(self, input_tsv_path: str, output_base_dir: str = "training_datasets", *, 
                 auto_start_oie: bool = True, silent_mode: bool = False):
        """Initialize DatasetController with comprehensive system validation.
        
        Args:
            input_tsv_path (str): Path to source TSV file containing raw dataset
            output_base_dir (str): Directory for output training datasets
            auto_start_oie (bool): Whether to automatically start OpenIE server
            silent_mode (bool): Suppress verbose logging in worker processes
        """
        self.input_tsv_path = input_tsv_path
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self._auto_start_oie = auto_start_oie  # Store OpenIE startup preference
        self._silent_mode = silent_mode  # Control logging verbosity
        
        # System memory validation
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 2:
            print(f"WARNING: Only {available_gb:.1f}GB RAM available. Consider freeing memory before processing large datasets.")
        
        # GPU memory assessment with DirectML support
        try:
            import torch
            
            # Check DirectML first (AMD GPUs)
            try:
                import torch_directml
                if torch_directml.is_available():
                    device_count = torch_directml.device_count()
                    device_name = torch_directml.device_name(0) if device_count > 0 else "Unknown"
                    print(f"DirectML GPU detected: {device_name} ({device_count} device(s))")
                    print("Note: DirectML memory monitoring limited")
            except ImportError:
                pass
            
            # Check CUDA GPUs
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_free = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"CUDA GPU: {gpu_memory:.1f}GB total, {gpu_free:.1f}GB reserved")
                if gpu_memory < 4:
                    print("WARNING: Low GPU memory. Consider using CPU for embeddings or reducing batch sizes.")
        except ImportError:
            pass
        
        # Initialize logging system with UTF-8 support
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_base_dir / f"adaptive_controller_run_{self.timestamp}.log"
        
        # Configure cross-platform logging
        import sys
        
        # Create formatters for consistent output
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        
        # Console handler for cross-platform compatibility
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        
        # Configure logger instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Show processing progress
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Define target IR models optimized for semantic chunking
        self.target_models = ["knrm", "drmm", "anmm", "dssm", "cdssm", "conv_knrm"]
        
        # Log system configuration details
        self._log_safe('info', f"Adaptive Controller initialized for dataset: {input_tsv_path}")
        self._log_safe('info', f"Target models: {', '.join(self.target_models)}")
        self._log_safe('info', f"Adaptive chunk range: {ADAPTIVE_CHUNKING_CONFIG['min_tokens']}-{ADAPTIVE_CHUNKING_CONFIG['max_tokens']} tokens")
        self._log_safe('info', f"Target: {ADAPTIVE_CHUNKING_CONFIG['target_tokens']} tokens ±{ADAPTIVE_CHUNKING_CONFIG['tolerance']*100:.0f}%")
        
        # Log current embedding model configuration
        current_model = COMMON_DEFAULTS["embedding_model"]
        model_info = None
        for key, info in AVAILABLE_EMBEDDING_MODELS.items():
            if info["model_name"] == current_model:
                model_info = info
                break
        
        if model_info:
            self._log_safe('info', f"Embedding Model: {current_model}")
            self._log_safe('info', f"VRAM Required: ~{model_info['vram_gb']}GB | Dimensions: {model_info['dimensions']}")
            # Display configuration in main process for immediate visibility
            if not self._silent_mode:
                print(f"[PROCESS] Using embedding model: {current_model}")
        else:
            self._log_safe('info', f"Embedding Model: {current_model} (custom model)")
            if not self._silent_mode:
                print(f"[PROCESS] Using embedding model: {current_model} (custom model)")
        
        # Validate OpenIE availability for relationship extraction
        self.oie_available = OIE_AVAILABLE_GLOBALLY
        if not self.oie_available:
            self._log_safe('warning', "OpenIE tool is not available. Configurations using OIE will run without relationship extraction.")
        else:
            self._log_safe('info', "OpenIE tool appears to be available for relationship extraction.")

        # Attempt to start OpenIE server if requested and not already running
        if self.oie_available and self._auto_start_oie:
            try:
                from Tool.OIE import _is_port_open
                if not _is_port_open(9000):
                    self._log_safe('info', "Starting OpenIE5 server on port 9000...")
                    # Initialize OIE to trigger auto-start mechanism
                    if extract_relations_from_paragraph is not None:  # type: ignore
                        extract_relations_from_paragraph("Connection test", port=9000, silent=True)  # type: ignore[arg-type]
                if _is_port_open(9000):
                    self._log_safe('info', "OpenIE5 server successfully started on port 9000.")
            except Exception:
                pass

    def _log_safe(self, level: str, message: str):
        """Safely log messages without emoji processing for cross-platform compatibility.
        
        Args:
            level (str): Logging level ('info', 'warning', 'error', 'debug')
            message (str): Message to log
        """
        if not self._silent_mode:  # Only log if not in silent mode
            getattr(self.logger, level.lower(), self.logger.info)(message)
    
    def _utility_memory_status(self) -> dict:
        """Monitor current system memory usage for processing optimization.
        
        Provides comprehensive memory statistics including system RAM, process memory,
        and GPU memory (if available) for performance monitoring and optimization.
        
        Returns:
            dict: Memory status containing:
                - total_ram_gb (float): Total system RAM in GB
                - available_ram_gb (float): Available RAM in GB
                - ram_usage_percent (float): System RAM utilization percentage
                - process_memory_mb (float): Current process memory usage in MB
                - gpu_memory_gb (float, optional): Total GPU memory if CUDA available
                - gpu_allocated_gb (float, optional): Allocated GPU memory if CUDA available
                - gpu_reserved_gb (float, optional): Reserved GPU memory if CUDA available
        """
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
            
            # Add GPU memory statistics if CUDA is available
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
        """Validate system memory safety before continuing processing operations.
        
        Monitors RAM usage and provides warnings when memory consumption approaches
        dangerous levels that could cause system instability or processing failures.
        
        Args:
            warn_threshold (float): RAM usage percentage threshold for warnings (default: 85.0)
            
        Returns:
            bool: True if memory usage is safe to continue, False if critical levels reached
        """
        status = self._utility_memory_status()
        if "error" in status:
            return True  # Cannot check memory status, assume safe to continue
            
        ram_usage = status.get("ram_usage_percent", 0)
        if ram_usage > warn_threshold:
            self._log_safe('warning', f"High RAM usage: {ram_usage:.1f}%. Consider reducing batch size or freeing memory.")
            if ram_usage > 95:
                self._log_safe('error', f"Critical RAM usage: {ram_usage:.1f}%. Processing may fail!")
                return False
        
        return True
    
    def _utility_display_embedding_models(self) -> None:
        """Display comprehensive list of available embedding models with system recommendations.
        
        Shows all configured embedding models with their specifications, memory requirements,
        and suitability recommendations based on current system capabilities.
        """
        
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
        """Change the active embedding model with comprehensive validation.
        
        Validates model availability, checks memory requirements, and updates the global
        configuration to use the specified embedding model for all operations.
        
        Args:
            model_key (str, optional): Key of the embedding model to activate.
                                     If None, displays available models.
        
        Returns:
            bool: True if model change was successful, False if validation failed
        """
        if model_key is None:
            self._utility_display_embedding_models()
            return True
            
        if model_key not in AVAILABLE_EMBEDDING_MODELS:
            available_keys = list(AVAILABLE_EMBEDDING_MODELS.keys())
            print(f"Invalid model key '{model_key}'. Available: {available_keys}")
            return False
        
        # Validate memory requirements against system capabilities
        model_info = AVAILABLE_EMBEDDING_MODELS[model_key]
        required_vram = model_info["vram_gb"]
        
        try:
            import torch
            
            # Check DirectML support first (AMD GPUs)
            try:
                import torch_directml
                if torch_directml.is_available():
                    print(f"DirectML GPU detected. Estimated requirement: ~{required_vram}GB VRAM")
                    # DirectML doesn't provide detailed memory info, so proceed with caution
                    if required_vram > 6:  # Conservative threshold for DirectML
                        print(f"WARNING: Model requires significant VRAM (~{required_vram}GB)")
                        confirm = input("Continue with DirectML? (y/n): ").strip().lower()
                        if confirm != 'y':
                            return False
            except ImportError:
                pass
            
            # Check CUDA GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if required_vram > gpu_memory_gb * 0.8:  # Leave 20% buffer for processing
                    print(f"WARNING: Model requires ~{required_vram}GB VRAM but only {gpu_memory_gb:.1f}GB available!")
                    print("Consider using CPU processing or a smaller embedding model.")
                    
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        return False
        except ImportError:
            print("GPU acceleration not available, will use CPU processing.")
        
        # Update the global model configuration
        old_model = COMMON_DEFAULTS["embedding_model"]
        COMMON_DEFAULTS["embedding_model"] = model_info["model_name"]
        
        print(f"Embedding model successfully changed:")
        print(f"   From: {old_model}")
        print(f"   To: {model_info['model_name']}")
        print(f"   VRAM: ~{required_vram}GB | Dimensions: {model_info['dimensions']}")
        
        return True
    
    def _utility_quick_model_switch(self) -> None:
        """Provide quick model switching interface with optimized presets.
        
        Displays common embedding model presets categorized by performance and memory
        requirements, allowing users to quickly select appropriate models for their system.
        """
        print("\nQUICK MODEL PRESETS:")
        print("1. High Quality (gte-large) - Best accuracy, requires 4.5GB+ VRAM")
        print("2. Balanced (gte-base) - Good quality, moderate memory (2GB VRAM)")
        print("3. Fast (minilm-l12) - Quick processing, low memory (0.7GB VRAM)")
        print("4. Ultra-Fast (minilm-l6) - Fastest processing, minimal memory (0.5GB VRAM)")
        print("5. Auto-select based on system capabilities")
        
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
                print("Failed to apply model preset.")
        elif choice:
            print("Invalid choice. Please select 1-5.")
    
    def helper_runtime_model_change(self, model_key: str = None) -> bool:
        """Enable runtime embedding model changes during processing operations.
        
        Allows dynamic switching of embedding models without restarting the controller,
        useful for testing different model configurations or adapting to system constraints.
        
        Args:
            model_key (str, optional): Key of the embedding model to switch to.
                                     If None, displays available options.
        
        Returns:
            bool: True if model change was successful, False otherwise
        """
        if model_key is None:
            print("Available embedding models:")
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
        """Execute memory-efficient chunking logic with comprehensive resource management.
        
        Processes DataFrame batches through specified chunking algorithms while maintaining
        strict memory controls and garbage collection to prevent system resource exhaustion.
        
        Memory Optimization Strategy:
            1. Embedding model: Fixed ~2-4GB VRAM (loaded once, persists across batches)
            2. Output accumulation: ~10-50MB per 1000 chunks (primary scaling concern)
            3. Temporary processing: ~1-10MB per row (cleared after each row)
            4. Linear scaling: Memory grows with output chunks, not input rows
        
        Args:
            df (pd.DataFrame): Input batch containing query, passage, and label columns
            config (dict): Chunking configuration containing:
                - method_choice (str): Chunking algorithm identifier ("1", "2", "3")
                - params (dict): Algorithm-specific parameters
                - include_oie (bool): Whether to use OpenIE relationship extraction
                - save_raw_oie (bool): Whether to preserve raw OIE output
        
        Returns:
            List[List[Any]]: Processed chunks as rows ready for dataset creation
        """
        output_rows = []
        method_choice = config["method_choice"]
        method_params = config["params"].copy()  # Use copy to prevent config mutation

        # Configure OpenIE integration based on availability and settings
        use_oie = config.get("include_oie", False) and self.oie_available
        method_params['include_oie'] = use_oie
        method_params['save_raw_oie'] = use_oie and config.get("save_raw_oie", False)

        # Map method choices to their corresponding chunking functions
        chunking_function_map = {
            "1": semantic_chunk_passage_from_grouping_logic,      # Semantic grouping
            "2": chunk_passage_text_splitter,                    # Semantic splitting
            "3": chunk_passage_text_splitter,                    # Text splitting
        }
        chunking_function = chunking_function_map.get(method_choice)

        if not chunking_function:
            self._log_safe('warning', f"No valid chunking function for method_choice '{method_choice}'. Skipping.")
            return []

        # Initialize memory monitoring for resource optimization
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # Convert to MB
        
        # Memory optimization: Clear buffer periodically to prevent accumulation
        max_buffer_chunks = 1000  # Buffer limit to maintain memory efficiency

        for row_idx, row_data in df.iterrows():
            try:
                query_text, passage_text, label = row_data.iloc[0], row_data.iloc[1], row_data.iloc[2]
                query_id = f"Q{row_idx}"
                passage_id = f"P{row_idx}"

                # Log processing progress at regular intervals for monitoring
                if row_idx % 10 == 0:
                    self._log_safe('info', f"[{config['name']}] Processing query_id: {query_id} (row {row_idx})")

                if not isinstance(passage_text, str) or not passage_text.strip():
                    self.logger.warning(f"Skipping row {row_idx} due to empty passage text.")
                    continue

                # Periodic memory monitoring and garbage collection
                if row_idx % 100 == 0:
                    current_memory = process.memory_info().rss / (1024**2)  # Convert to MB
                    memory_growth = current_memory - initial_memory
                    if memory_growth > 1000:  # Alert if memory grew by more than 1GB
                        self._log_safe('warning', f"High memory usage detected: {memory_growth:.1f}MB growth. Running garbage collection...")
                        gc.collect()  # Force garbage collection to free memory

                # Prepare parameters for the selected chunking function
                common_params = {
                    'passage_text': passage_text,
                    'doc_id': f"{query_id}_{passage_id}",
                    'output_dir': str(self.output_base_dir / config["name"]),
                    'silent': True,
                    **method_params
                }
                
                # Add semantic-specific parameters for methods 1 and 2
                if method_choice in ["1", "2"]:  # Semantic grouping and semantic splitting
                    common_params['embedding_model'] = COMMON_DEFAULTS["embedding_model"]
                    common_params['device'] = COMMON_DEFAULTS["device_preference"]

                # Execute the selected chunking algorithm
                chunked_tuples: List[Tuple[str, str, Optional[str]]] = chunking_function(**common_params)

                if not chunked_tuples:
                    self.logger.warning(f"No chunks generated for row {row_idx}. Original passage: {passage_text[:100]}...")
                    continue
                
                # Format output rows for dataset creation
                for gen_chunk_id, chunk_text, oie_string in chunked_tuples:
                    output_rows.append([
                        query_id, query_text, passage_id, passage_text, label,
                        gen_chunk_id, chunk_text, oie_string if oie_string else ""
                    ])
                    
                # Memory optimization: Clear buffer periodically to prevent accumulation
                # Memory optimization: Clear buffer when reaching capacity
                if len(output_rows) >= max_buffer_chunks:
                    self._log_safe('debug', f"Buffer size reached {len(output_rows)} chunks, triggering memory cleanup...")
                    # Note: In production environments, would flush to temporary file here
                    # For now, trigger aggressive garbage collection to free memory
                    gc.collect()
                    
                # Immediately free memory from processed chunks
                del chunked_tuples
                
            except Exception as e:
                self.logger.error(f"Error processing row {row_idx} in config '{config['name']}': {e}", exc_info=False)
        
        # Final memory cleanup and statistics
        gc.collect()
        final_memory = process.memory_info().rss / (1024**2)  # Convert to MB
        self._log_safe('info', f"Batch processing completed. Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB ({len(output_rows)} chunks)")
        
        return output_rows

    def _process_batches_parallel(self, batches: List[Tuple[int, pd.DataFrame]], config: dict, 
                                 config_workers: int, task_name: str) -> List[List[List[Any]]]:
        """Process multiple batches in parallel for optimal resource utilization.
        
        Distributes DataFrame batches across multiple worker processes to maximize
        throughput while maintaining memory efficiency and system stability.
        
        Args:
            batches (List[Tuple[int, pd.DataFrame]]): List of (batch_number, dataframe) tuples to process
            config (dict): Chunking configuration containing method and parameters
            config_workers (int): Number of worker processes to allocate for this configuration
            task_name (str): Descriptive name of the processing task for logging
            
        Returns:
            List[List[List[Any]]]: Nested list containing chunked rows for each processed batch
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        self._log_safe('info', f"Processing {len(batches)} batches with {config_workers} workers for {task_name}")
        
        # Optimize worker allocation based on available resources
        effective_workers = min(config_workers, len(batches), multiprocessing.cpu_count())
        self._log_safe('info', f"Using {effective_workers} effective workers for {len(batches)} batches")
        
        # Get current embedding model for workers
        current_embedding_model = COMMON_DEFAULTS["embedding_model"]
        self._log_safe('info', f"Each worker will use dedicated embedding model: {current_embedding_model}")
        
        # Store results in order
        batch_results = [None] * len(batches)
        
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all batches with dedicated embedding model instances for each worker
            future_to_batch = {}
            worker_counter = 0
            
            for batch_num, batch_df in batches:
                worker_id = worker_counter % effective_workers + 1  # Assign unique worker ID (1, 2, 3...)
                future = executor.submit(
                    helper_process_single_batch, 
                    batch_df, 
                    config, 
                    batch_num,
                    embedding_model=current_embedding_model,  # Provide model to each worker
                    worker_id=worker_id
                )
                future_to_batch[future] = batch_num - 1
                worker_counter += 1
            
            # Collect results in completion order while maintaining batch indexing
            completed_count = 0
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                completed_count += 1
                
                try:
                    chunked_rows = future.result()
                    batch_results[batch_index] = chunked_rows
                    chunk_count = len(chunked_rows) if chunked_rows else 0
                    self._log_safe('info', f"Batch {batch_index + 1} completed: {chunk_count} chunks ({completed_count}/{len(batches)})")
                except Exception as e:
                    self._log_safe('error', f"Batch {batch_index + 1} failed: {e}")
                    batch_results[batch_index] = []
        
        # Return results maintaining original batch order
        return batch_results

    def _run_single_config(self, config: dict, ranking_workers: Optional[int] = None, 
                          config_workers: Optional[int] = None) -> Tuple[bool, str, Optional[float]]:
        """Execute single chunking configuration with comprehensive resource management.
        
        Processes complete dataset through specified chunking algorithm, handling batch
        processing, memory optimization, and performance monitoring with detailed logging.
        
        Args:
            config (dict): Complete configuration dictionary containing:
                - name (str): Configuration identifier for output naming
                - description (str): Human-readable description of chunking strategy
                - method_choice (str): Algorithm selector ("1", "2", "3")
                - params (dict): Algorithm-specific parameters
                - include_oie (bool): Whether to use OpenIE relationship extraction
            ranking_workers (Optional[int]): Number of workers for ranking phase.
                                           None triggers automatic calculation based on system resources.
            config_workers (Optional[int]): Number of workers for this specific configuration.
                                           None enables sequential processing for memory efficiency.
        
        Returns:
            Tuple[bool, str, Optional[float]]: (success_status, output_path, processing_time)
        """
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
            # Collect all batches first if using parallel processing
            if config_workers and config_workers > 1:
                self._log_safe('info', f"Using {config_workers} workers for config {task_name}")
                batches = []
                with pd.read_csv(self.input_tsv_path, sep='\t', engine='python', chunksize=PROCESSING_BATCH_SIZE, header=0, on_bad_lines='warn') as reader:
                    for i, batch_df in enumerate(reader, 1):
                        batches.append((i, batch_df))
                
                # Execute parallel batch processing for optimal performance
                all_chunked_rows = self._process_batches_parallel(batches, config, config_workers, task_name)
                
                # Consolidate and write all processed results to output file
                total_chunks = 0
                with open(final_output_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(['query_id', 'query_text', 'passage_id', 'original_passage', 'label', 'chunk_id', 'chunk_text', 'raw_oie_data'])
                    for batch_chunks in all_chunked_rows:
                        if batch_chunks:
                            writer.writerows(batch_chunks)
                            total_chunks += len(batch_chunks)
                
                self._log_safe('info', f"Parallel processing completed successfully. Total chunks generated: {total_chunks}")
            else:
                # Sequential processing for memory-constrained environments
                with pd.read_csv(self.input_tsv_path, sep='\t', engine='python', chunksize=PROCESSING_BATCH_SIZE, header=0, on_bad_lines='warn') as reader:
                    for i, batch_df in enumerate(reader, 1):
                        batch_start_row = (i-1) * PROCESSING_BATCH_SIZE
                        batch_end_row = batch_start_row + len(batch_df) - 1
                        self._log_safe('info', f"--- Processing Batch {i} for {task_name} (rows {batch_start_row}-{batch_end_row}) ---")
                        
                        # Clear cached data to minimize memory footprint
                        gc.collect()
                        
                        chunked_rows = self.process_execute_chunking_logic(batch_df, config)
                        batch_chunks_count = len(chunked_rows)
                        total_chunks += batch_chunks_count

                        # Immediately write to disk and clear from memory to prevent accumulation
                        write_mode = 'w' if is_first_batch else 'a'
                        with open(final_output_file, write_mode, encoding='utf-8', newline='') as f:
                            writer = csv.writer(f, delimiter='\t')
                            if is_first_batch:
                                writer.writerow(['query_id', 'query_text', 'passage_id', 'original_passage', 'label', 'chunk_id', 'chunk_text', 'raw_oie_data'])
                                is_first_batch = False
                            if chunked_rows:
                                writer.writerows(chunked_rows)
                        
                        # Aggressively clear batch data from memory to maintain efficiency
                        del chunked_rows, batch_df
                        gc.collect()
                        
                        self._log_safe('info', f"Batch {i} completed. Generated {batch_chunks_count} chunks. Total chunks so far: {total_chunks}")
                        
                        # Monitor memory usage and provide warnings
                        import psutil
                        memory_usage = psutil.virtual_memory().percent
                        if memory_usage > 85:
                            self._log_safe('warning', f"High RAM usage: {memory_usage:.1f}%. Consider reducing batch size to prevent system instability.")
                
                self._log_safe('info', f"Total chunks generated for {task_name}: {total_chunks}")

        except Exception as e:
            self._log_safe('error', f"Critical error occurred during batch processing for {task_name}: {e}")
            batch_success = False

        if not batch_success or not os.path.exists(final_output_file):
            return False, "", None

        # Skip detailed analysis/statistics to reduce logging overhead and improve performance
        final_compliance = None

        # Execute ranking and filtering phase to optimize chunk quality
        try:
            # Retrieve percentile thresholds from environment variables with fallback defaults
            try:
                up_perc = int(os.getenv("UPPER_PERCENTILE", "80"))
                low_perc = int(os.getenv("LOWER_PERCENTILE", "20"))
            except ValueError:
                up_perc, low_perc = 80, 20

            # Prepare system for ranking phase with memory optimization
            gc.collect()
            self._log_safe('info', f"Starting ranking phase for {task_name}")
            
            ranked_filtered_path = self.helper_rank_and_filter_chunks(str(final_output_file), config_output_dir,
                                                               upper_percentile=up_perc, lower_percentile=low_perc,
                                                               ranking_workers=ranking_workers)
            if ranked_filtered_path:
                self._log_safe('info', f"Ranked & filtered results saved to: {ranked_filtered_path}")
            
            # Aggressive memory cleanup after ranking operations
            gc.collect()
            
        except Exception as e:
            self._log_safe('error', f"Ranking/filtering failed for {task_name}: {e}")
            # Continue processing even if ranking fails to maintain pipeline robustness
            import traceback
            self._log_safe('debug', f"Ranking error details: {traceback.format_exc()}")
        
        return True, str(final_output_file), final_compliance
    
    def helper_rank_and_filter_chunks(self, chunks_tsv: str, output_dir: Path,
                                upper_percentile: int = 80, lower_percentile: int = 20, 
                                ranking_workers: Optional[int] = None) -> str:
        """Rank chunks using RRF algorithm and filter by percentile thresholds - OPTIMIZED VERSION.
        
        Implements Reciprocal Rank Fusion (RRF) to rank chunks based on semantic similarity
        and relevance scores, then filters to retain only top and bottom quartiles for
        balanced training data quality.

        Args:
            chunks_tsv (str): Path to input TSV file containing chunked data
            output_dir (Path): Directory for output files
            upper_percentile (int): Upper percentile threshold for high-quality chunks (default: 80)
            lower_percentile (int): Lower percentile threshold for low-quality chunks (default: 20)
            ranking_workers (Optional[int]): Number of workers for parallel ranking.
                                           None triggers automatic calculation.

        Returns:
            str: Path to ranked and filtered output file, or empty string on failure
        """
        try:
            # Calculate optimal ranking workers if not specified
            if ranking_workers is None:
                ranking_workers = _utility_calculate_optimal_ranking_workers()
                self._log_safe('info', f"Auto-calculated ranking workers: {ranking_workers}")
            else:
                self._log_safe('info', f"Using specified ranking workers: {ranking_workers}")
            
            # Import optimized ranking function
            from Tool.rank_chunks_optimized import rank_and_filter_chunks_optimized
            
            self._log_safe('info', f"Starting optimized ranking for {chunks_tsv}")
            try:
                # Try with max_workers parameter first
                result_path = rank_and_filter_chunks_optimized(
                    chunks_tsv=chunks_tsv,
                    output_dir=output_dir, 
                    upper_percentile=upper_percentile,
                    lower_percentile=lower_percentile,
                    model_name=COMMON_DEFAULTS["embedding_model"],
                    max_workers=ranking_workers  # Pass ranking workers to optimization function
                )
            except TypeError:
                # Fallback: call without max_workers parameter
                self._log_safe('warning', f"Optimized ranking function doesn't support max_workers parameter, using fallback")
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
            return self._helper_rank_and_filter_chunks_fallback(chunks_tsv, output_dir, upper_percentile, lower_percentile, ranking_workers)
        except Exception as e:
            self._log_safe('error', f"Optimized ranking failed: {e}")
            return ""
    
    def _helper_rank_and_filter_chunks_fallback(self, chunks_tsv: str, output_dir: Path,
                                upper_percentile: int = 80, lower_percentile: int = 20, 
                                ranking_workers: Optional[int] = None) -> str:
        """Fallback to original ranking implementation
        
        Args:
            chunks_tsv: Path to chunks TSV file
            output_dir: Output directory
            upper_percentile: Upper percentile for positive samples
            lower_percentile: Lower percentile for negative samples
            ranking_workers: Number of workers for ranking (unused in fallback, kept for compatibility)
        """
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
    def data_create_controller_main(self, use_parallel: bool = False, max_workers: Optional[int] = None, 
                                   ranking_workers: Optional[int] = None, config_workers: Optional[int] = None) -> Dict[str, Any]:
        """Create all 6 adaptive datasets with simple re-chunking loop
        
        Args:
            use_parallel: Whether to use parallel processing for chunking
            max_workers: Number of workers for chunking/OIE phase (None = auto-calculate)
            ranking_workers: Number of workers for ranking phase (None = auto-calculate)
            config_workers: Number of workers per config for internal parallelism (None = sequential)
        """
        
        if use_parallel:
            return self.process_create_all_datasets_parallel(max_workers, ranking_workers, config_workers)
        else:
            return self.process_create_all_datasets_sequential(ranking_workers, config_workers)

    def process_create_all_datasets_sequential(self, ranking_workers: Optional[int] = None, 
                                              config_workers: Optional[int] = None) -> Dict[str, Any]:
        """Execute sequential dataset creation with comprehensive configuration processing.
        
        Processes all configured chunking algorithms sequentially, providing detailed
        progress tracking and resource optimization for memory-constrained environments.
        
        Args:
            ranking_workers (Optional[int]): Number of workers for ranking phase.
                                           None triggers automatic calculation based on system resources.
            config_workers (Optional[int]): Number of workers per configuration for internal parallelism.
                                           None enables sequential batch processing for maximum memory efficiency.
        
        Returns:
            Dict[str, Any]: Processing results containing:
                - success (bool): Overall processing success status
                - results (List[Dict]): Individual configuration results with timing and output paths
                - successful_tasks (int): Number of configurations completed successfully
                - total_tasks (int): Total number of configurations processed
        """
        self._log_safe('info', "="*80)
        self._log_safe('info', "STARTING SEQUENTIAL ADAPTIVE DATASET CREATION")
        self._log_safe('info', "="*80)
        
        if ranking_workers is None:
            ranking_workers = _utility_calculate_optimal_ranking_workers()
            self._log_safe('info', f"Auto-calculated ranking workers: {ranking_workers}")
        else:
            self._log_safe('info', f"Using specified ranking workers: {ranking_workers}")
        
        if not self._utility_validate_input_file():
            self._log_safe('error', "Input file validation failed. Aborting sequential processing.")
            return {"success": False, "results": []}
        
        # Filter configurations to exclude OpenIE-dependent tasks for standard processing
        configs_to_run = [c for c in RUN_CONFIGURATIONS if not c.get("include_oie", False)]
        
        results = []
        successful_tasks = 0
        
        if config_workers and config_workers > 1:
            self._log_safe('info', f"Each configuration will use {config_workers} workers for batch processing")
        else:
            self._log_safe('info', "Using sequential batch processing for each configuration (maximum memory efficiency)")
        
        for i, config in enumerate(configs_to_run, 1):
            self._log_safe('info', f"\n>>> Configuration {i}/{len(configs_to_run)}: {config['name']} <<<")
            start_time = time.time()
            
            success, output_file, final_compliance = self._run_single_config(config, ranking_workers=ranking_workers, config_workers=config_workers)

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

    def process_create_all_datasets_parallel(self, max_workers: Optional[int] = None, 
                                            ranking_workers: Optional[int] = None, 
                                            config_workers: Optional[int] = None) -> Dict[str, Any]:
        """Execute parallel dataset creation using optimized ProcessPoolExecutor.
        
        Leverages multi-core processing for maximum throughput, automatically calculating
        optimal worker allocation based on system resources and memory constraints.
        
        Args:
            max_workers (Optional[int]): Number of workers for chunking/OIE phase.
                                       None triggers automatic calculation based on memory and CPU.
            ranking_workers (Optional[int]): Number of workers for ranking phase.
                                            None triggers automatic calculation.
            config_workers (Optional[int]): Number of workers per configuration for internal parallelism.
                                           None enables sequential batch processing.
        
        Returns:
            Dict[str, Any]: Processing results containing:
                - success (bool): Overall processing success status
                - results (List[Dict]): Individual configuration results with timing and output paths
                - successful_tasks (int): Number of configurations completed successfully
                - total_tasks (int): Total number of configurations processed
                - worker_allocation (Dict): Details about worker resource allocation
        """
        self.logger.info("="*80)
        self.logger.info("STARTING PARALLEL ADAPTIVE DATASET CREATION")
        self.logger.info("="*80)
        
        # Import psutil for comprehensive memory monitoring
        import psutil
        
        if max_workers is None:
            # Calculate optimal chunking workers based on memory and CPU availability
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            memory_per_worker = 2.0  # GB required for gte-base model (chunking phase)
            max_by_memory = max(1, int((available_ram_gb * 0.7) / memory_per_worker))
            max_workers = min(multiprocessing.cpu_count(), max_by_memory)
        
        if ranking_workers is None:
            ranking_workers = _utility_calculate_optimal_ranking_workers()
            
        self.logger.info(f"Using up to {max_workers} parallel workers for chunking/OIE operations")
        self.logger.info(f"Using up to {ranking_workers} parallel workers for ranking operations")
        self.logger.info(f"Memory consideration: {psutil.virtual_memory().available / (1024**3):.1f}GB available RAM")
        
        if not self._utility_validate_input_file():
            self.logger.error("Input file validation failed. Aborting parallel processing.")
            return {"success": False, "results": []}
        
        # Filter to non-OpenIE configurations for standard parallel processing
        non_oie_configs = [c for c in RUN_CONFIGURATIONS if not c.get("include_oie")]
        
        # Analyze worker efficiency and provide optimization recommendations
        actual_parallel_tasks = len(non_oie_configs)
        if max_workers > actual_parallel_tasks:
            self.logger.warning(f"WARNING: {max_workers} workers requested but only {actual_parallel_tasks} tasks available!")
            self.logger.warning(f"Only {actual_parallel_tasks} processes will run simultaneously.")
            self.logger.warning(f"Consider using sequential mode or enabling two-phase processing for better worker utilization.")
        
        self.logger.info(f"Parallel execution plan: {actual_parallel_tasks} tasks across {min(max_workers, actual_parallel_tasks)} workers")
        
        results = []
        successful_tasks = 0
        
        # Execute non-OpenIE configurations with optimized parallel processing
        if non_oie_configs:
            self.logger.info(f"Running {len(non_oie_configs)} non-OIE configurations in parallel...")
            # Use optimal worker count based on available tasks
            effective_workers = min(max_workers, len(non_oie_configs))
            self.logger.info(f"Using {effective_workers} effective workers for {len(non_oie_configs)} tasks")
            
            # Log internal worker configuration details
            if config_workers and config_workers > 1:
                self.logger.info(f"Each configuration will use {config_workers} internal workers for batch processing")
                total_workers = effective_workers * config_workers
                self.logger.info(f"Total system workers: {total_workers} (configs: {effective_workers} × workers per config: {config_workers})")
            else:
                self.logger.info("Each configuration will use sequential batch processing")
            
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                # Log initial GPU status
                initial_gpu = _utility_monitor_gpu_usage()
                if "error" not in initial_gpu:
                    gpu_type = initial_gpu.get("type", "Unknown")
                    if gpu_type == "DirectML":
                        self.logger.info(f"Initial GPU status - DirectML: {initial_gpu['device_count']} device(s) available")
                        self.logger.info(f"   Device: {initial_gpu['device_name']}")
                    elif gpu_type == "CUDA":
                        self.logger.info("Initial GPU status - CUDA:")
                        for gpu_id, stats in initial_gpu.items():
                            if gpu_id.startswith("gpu_"):
                                self.logger.info(f"   {gpu_id}: {stats['allocated_gb']:.1f}GB allocated / {stats['total_gb']:.1f}GB total")
                    else:
                        self.logger.info(f"Initial GPU status: {initial_gpu}")
                else:
                    self.logger.warning(f"GPU monitoring unavailable: {initial_gpu.get('error', 'Unknown error')}")
                
                future_to_config = {
                    executor.submit(helper_run_chunking_config_process, config, self.input_tsv_path, 
                                  str(self.output_base_dir), auto_start_oie=False, 
                                  embedding_model=COMMON_DEFAULTS["embedding_model"], 
                                  ranking_workers=ranking_workers, config_workers=config_workers): config
                    for config in non_oie_configs
                }
                
                # Monitor GPU during execution
                completed_count = 0
                total_tasks = len(future_to_config)
                
                for future in as_completed(future_to_config):
                    config_name = future_to_config[future]['name']
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                        if result["success"]: 
                            successful_tasks += 1
                            self.logger.info(f"Completed {config_name} ({completed_count}/{total_tasks})")
                        else:
                            self.logger.error(f"Failed {config_name} ({completed_count}/{total_tasks})")
                    except Exception as e:
                        self.logger.error(f"Process for {config_name} generated an exception: {e}")
                        results.append({"config_name": config_name, "success": False, "error": str(e)})
                    
                    # Log GPU status every task completion  
                    if completed_count % 1 == 0:  # Log after each task 
                        current_gpu = _utility_monitor_gpu_usage()
                        if "error" not in current_gpu:
                            gpu_type = current_gpu.get("type", "Unknown")
                            if gpu_type == "DirectML":
                                self.logger.info(f"GPU status after task {completed_count}/{total_tasks} - DirectML active")
                            elif gpu_type == "CUDA":
                                self.logger.info(f"GPU status after task {completed_count}/{total_tasks}:")
                                for gpu_id, stats in current_gpu.items():
                                    if gpu_id.startswith("gpu_"):
                                        self.logger.info(f"   {gpu_id}: {stats['allocated_gb']:.1f}GB allocated ({stats['utilization_percent']:.1f}% utilization)")
                
                # Final GPU status
                final_gpu = _utility_monitor_gpu_usage()
                if "error" not in final_gpu:
                    gpu_type = final_gpu.get("type", "Unknown")
                    if gpu_type == "DirectML":
                        self.logger.info("Final GPU status - DirectML: Processing completed successfully")
                    elif gpu_type == "CUDA":
                        self.logger.info("Final GPU status - CUDA:")
                        for gpu_id, stats in final_gpu.items():
                            if gpu_id.startswith("gpu_"):
                                self.logger.info(f"   {gpu_id}: {stats['allocated_gb']:.1f}GB allocated / {stats['total_gb']:.1f}GB total")
                    else:
                        self.logger.info(f"Final GPU status: {final_gpu}")

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
    # NEW: Helper để convert existing chunks sang OIE version
    # ============================================================
    def convert_existing_chunks_to_oie(self, source_chunks_file: str, target_config_name: str = None) -> str:
        """Convert existing chunk file to OIE-enhanced version
        
        Args:
            source_chunks_file: Path to existing chunks TSV file (output from without_OIE config)
            target_config_name: Target config name (e.g., 'semantic_splitter_with_OIE')
            
        Returns:
            Path to new OIE-enhanced file
        """
        try:
            from pathlib import Path as _Path
            
            if not os.path.exists(source_chunks_file):
                self._log_safe('error', f'Source chunks file not found: {source_chunks_file}')
                return ""
            
            # Auto-detect target config name if not provided
            source_path = _Path(source_chunks_file)
            if target_config_name is None:
                # Convert 'semantic_splitter_without_OIE' -> 'semantic_splitter_with_OIE'
                source_dir_name = source_path.parent.name
                if source_dir_name.endswith('_without_OIE'):
                    target_config_name = source_dir_name.replace('_without_OIE', '_with_OIE')
                else:
                    target_config_name = f"{source_dir_name}_with_OIE"
            
            # Create target directory
            target_dir = self.output_base_dir / target_config_name
            target_dir.mkdir(exist_ok=True)
            
            target_file = target_dir / f"output_chunks_oie_converted_{self.timestamp}.tsv"
            
            self._log_safe('info', f'Converting {source_chunks_file} to OIE version: {target_file}')
            
            # Use the existing OIE augmentation function
            result_path = self.process_augment_with_oie(source_chunks_file, threads=8)
            
            if result_path:
                # Copy to target location with proper naming
                import shutil
                shutil.copy2(result_path, target_file)
                self._log_safe('info', f'OIE conversion completed: {target_file}')
                return str(target_file)
            else:
                self._log_safe('error', 'OIE conversion failed')
                return ""
                
        except Exception as e:
            self._log_safe('error', f'Error converting chunks to OIE: {e}')
            return ""

    # ============================================================
    # NEW: Helper để tự động tạo cặp without/with OIE từ 1 config
    # ============================================================
    def process_config_pair_without_with_oie(self, base_config_name: str, 
                                           ranking_workers: Optional[int] = None,
                                           config_workers: Optional[int] = None) -> dict:
        """Process a config pair: first without OIE, then convert to with OIE
        
        Args:
            base_config_name: Base config name (e.g., 'semantic_splitter')
            ranking_workers: Number of ranking workers
            config_workers: Number of config workers
            
        Returns:
            Dictionary with results for both versions
        """
        try:
            # Find the without_OIE config
            without_oie_config = None
            for cfg in RUN_CONFIGURATIONS:
                if cfg['name'] == f"{base_config_name}_without_OIE":
                    without_oie_config = cfg
                    break
            
            if not without_oie_config:
                self._log_safe('error', f'Config {base_config_name}_without_OIE not found')
                return {"success": False, "without_oie": None, "with_oie": None}
            
            results = {}
            
            # Step 1: Run without_OIE config
            self._log_safe('info', f'Step 1: Running {base_config_name}_without_OIE')
            success_without, output_file_without, compliance_without = self._run_single_config(
                without_oie_config, 
                ranking_workers=ranking_workers, 
                config_workers=config_workers
            )
            
            results['without_oie'] = {
                'success': success_without,
                'output_file': output_file_without,
                'compliance': compliance_without
            }
            
            if not success_without:
                self._log_safe('error', f'Failed to run {base_config_name}_without_OIE')
                return {"success": False, **results}
            
            # Step 2: Convert to with_OIE version
            self._log_safe('info', f'Step 2: Converting to {base_config_name}_with_OIE')
            oie_output_file = self.convert_existing_chunks_to_oie(
                output_file_without, 
                f"{base_config_name}_with_OIE"
            )
            
            results['with_oie'] = {
                'success': bool(oie_output_file),
                'output_file': oie_output_file,
                'source_file': output_file_without
            }
            
            if oie_output_file:
                self._log_safe('info', f'Successfully created {base_config_name} pair')
                return {"success": True, **results}
            else:
                self._log_safe('error', f'Failed to convert to {base_config_name}_with_OIE')
                return {"success": False, **results}
                
        except Exception as e:
            self._log_safe('error', f'Error processing config pair {base_config_name}: {e}')
            return {"success": False, "error": str(e)}

    # ============================================================
    # OpenIE Integration: Augment existing datasets with relationship extraction
    # ============================================================
    def process_augment_with_oie(self, source_tsv: str, threads: int = 8, chunk_rows: int = 1000) -> str:
        """Augment existing TSV datasets with OpenIE relationship extraction in parallel.
        
        Processes existing chunked datasets to add OpenIE-based relationship extraction,
        enabling enhanced semantic understanding for information retrieval tasks.
        
        Args:
            source_tsv (str): Path to source TSV file containing chunked data
            threads (int): Number of parallel threads for OIE processing (default: 8)
            chunk_rows (int): Number of rows to process per chunk for memory efficiency (default: 1000)
        
        Returns:
            str: Path to augmented output file with OIE data, or empty string on failure
        
        Note:
            Creates sibling directory with "_with_OIE" suffix for organized output structure.
            Preserves original data while adding 'raw_oie_data' column with extracted relationships.
        """
        try:
            if not self.oie_available:
                self._log_safe('error', 'OpenIE tool not available for relationship extraction.')
                return ""

            import pandas as _pd
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from pathlib import Path as _Path
            from Method.Text_Splitter import format_oie_triples_to_string  # Use shared formatting function

            if not os.path.exists(source_tsv):
                self._log_safe('error', f'Source file not found for augmentation: {source_tsv}')
                return ""

            # Determine output directory based on source configuration (create sibling _with_OIE)
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
                """Enrich chunk with OpenIE relationship data while preserving original structure."""
                # Ensure raw_oie_data column exists for relationship storage
                if 'raw_oie_data' not in chunk_df.columns:
                    chunk_df['raw_oie_data'] = ''
                if chunk_df['raw_oie_data'].dtype != 'O':
                    chunk_df['raw_oie_data'] = chunk_df['raw_oie_data'].astype(object)

                # Identify rows requiring OpenIE processing
                todo_idx_local = chunk_df.index[chunk_df['raw_oie_data'].isna() | (chunk_df['raw_oie_data'] == '')].tolist()
                if not todo_idx_local:
                    return chunk_df  # No processing needed

                self._log_safe('info', f"[augment_with_oie] Processing chunk size={len(chunk_df)} | OIE needed for {len(todo_idx_local)} rows")

                def _process_local(local_idx):
                    """Process individual row for OpenIE relationship extraction."""
                    try:
                        qid = str(chunk_df.at[local_idx, 'query_id']) if 'query_id' in chunk_df.columns else 'N/A'
                        cid = str(chunk_df.at[local_idx, 'chunk_id']) if 'chunk_id' in chunk_df.columns else f'row{local_idx}'
                        # Log every 10th OIE processing to prevent log spam
                        if local_idx % 10 == 0:
                            self._log_safe('info', f"[augment_with_oie] Processing query_id={qid} | chunk_id={cid}")
                    except Exception:
                        pass

                    txt = str(chunk_df.at[local_idx, 'chunk_text'])
                    triples = extract_relations_from_paragraph(txt, silent=True)  # type: ignore[arg-type]
                    return local_idx, (format_oie_triples_to_string(triples) if triples else '')

                # Execute parallel OpenIE processing for efficient throughput
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    fut_map = {executor.submit(_process_local, li): li for li in todo_idx_local}
                    for fut in as_completed(fut_map):
                        idx_local, oie_str_local = fut.result()
                        chunk_df.at[idx_local, 'raw_oie_data'] = oie_str_local

                return chunk_df

            # Process each chunk incrementally and write to output for memory efficiency
            for chunk_df in reader_iter:
                if 'chunk_text' not in chunk_df.columns:
                    self._log_safe('error', f"Required 'chunk_text' column missing in chunk; aborting augmentation.")
                    return ""

                processed_chunk = enrich_chunk(chunk_df)
                processed_chunk.to_csv(out_path, sep='\t', index=False, mode='w' if first_chunk else 'a', header=first_chunk)
                first_chunk = False

            # Generate simplified 3-column version for compatibility
            try:
                simple_df = _pd.read_csv(out_path, sep='\t', usecols=['query_text', 'chunk_text', 'label'], on_bad_lines='skip', engine='python')  # type: ignore[arg-type]
                simple_df.columns = ['query', 'passage', 'label']
                simple_df.to_csv(out_path.with_name(out_path.stem + '_3col.tsv'), sep='\t', index=False)
            except Exception:
                pass

            # Apply RRF filtering to augmented dataset for quality optimization
            try:
                try:
                    up_perc = int(os.getenv("UPPER_PERCENTILE", "80"))
                    low_perc = int(os.getenv("LOWER_PERCENTILE", "20"))
                except ValueError:
                    up_perc, low_perc = 80, 20

                filtered_path = self.helper_rank_and_filter_chunks(str(out_path), out_path.parent, upper_percentile=up_perc, lower_percentile=low_perc)
                if filtered_path:
                    self._log_safe('info', f"RRF filtered file saved to: {filtered_path}")
            except Exception as e:
                self._log_safe('error', f"RRF filtering failed for augmented file {out_path}: {e}")

            self._log_safe('info', f'Augmented dataset saved successfully: {out_path}')
            return str(out_path)
        except Exception as e:
            self._log_safe('error', f'OpenIE augmentation failed: {e}')
            return ""

    # ============================================================
    # Two-Phase Processing: Chunking followed by OpenIE augmentation
    # ============================================================
    def process_create_all_datasets_two_phase(self, *, non_oie_parallel: bool = True, max_workers: Optional[int] = None, 
                                             oie_threads: int = 8, ranking_workers: Optional[int] = None):
        """Execute two-phase dataset creation: (1) chunking without OIE, (2) OpenIE augmentation.
        
        Optimizes resource utilization by separating computationally intensive chunking
        from I/O-intensive OpenIE processing, enabling better parallelization strategies.
        
        Args:
            non_oie_parallel (bool): Whether to use parallel processing for chunking phase (default: True)
            max_workers (Optional[int]): Number of workers for chunking phase.
                                       None triggers automatic calculation.
            oie_threads (int): Number of threads for OpenIE processing phase (default: 8)
            ranking_workers (Optional[int]): Number of workers for ranking phase.
                                            None triggers automatic calculation.
        
        Returns:
            Dict[str, Any]: Comprehensive processing results containing both phases
        """

        # ---------------- PHASE 1: Chunking without OpenIE ----------------
        phase1_controller = self  

        configs_phase1 = [c for c in RUN_CONFIGURATIONS if not c.get('include_oie')]
        if not configs_phase1:
            self._log_safe('error', 'No without_OIE configurations found for processing.')
            return

        results_phase1 = []
        succ1 = 0

        if non_oie_parallel:
            if max_workers is None:
                # Calculate chunking workers
                import psutil
                available_ram_gb = psutil.virtual_memory().available / (1024**3)
                memory_per_worker = 2.0  # GB for gte-base model  
                max_by_memory = max(1, int((available_ram_gb * 0.7) / memory_per_worker))
                max_workers = min(multiprocessing.cpu_count(), max_by_memory)
            
            if ranking_workers is None:
                ranking_workers = _utility_calculate_optimal_ranking_workers()
                
            self._log_safe('info', f'Pha 1: chạy song song (chunking_workers={max_workers}, ranking_workers={ranking_workers})')
            self._log_safe('info', f'Memory consideration: {psutil.virtual_memory().available / (1024**3):.1f}GB available RAM')

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                fut_to_cfg = {executor.submit(helper_run_chunking_config_process, cfg, phase1_controller.input_tsv_path, str(phase1_controller.output_base_dir), auto_start_oie=False, embedding_model=COMMON_DEFAULTS["embedding_model"], ranking_workers=ranking_workers): cfg for cfg in configs_phase1}
                for fut in as_completed(fut_to_cfg):
                    result = fut.result()
                    results_phase1.append(result)
                    if result.get('success'):
                        succ1 += 1
        else:
            if ranking_workers is None:
                ranking_workers = _utility_calculate_optimal_ranking_workers()
                
            self._log_safe('info', f'Pha 1: chạy tuần tự (ranking_workers={ranking_workers})')
            for cfg in configs_phase1:
                success, out_file, _ = phase1_controller._run_single_config(cfg, ranking_workers=ranking_workers)
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
def helper_run_chunking_config_process(config: dict, input_tsv_path: str, output_base_dir: str, *, 
                                     auto_start_oie: bool = True, embedding_model: str = None, 
                                     ranking_workers: Optional[int] = None, 
                                     config_workers: Optional[int] = None) -> dict:
    """Process-based worker function to run a single chunking configuration.

    auto_start_oie:
        Quyết định có tự khởi động server OIE5 trong tiến trình con hay không.
        Đối với giai đoạn without_OIE nên đặt False để tránh khởi động thừa.
    embedding_model:
        Model embedding để sử dụng, ghi đè COMMON_DEFAULTS nếu được cung cấp.
    ranking_workers:
        Số worker cho giai đoạn ranking (None = auto-calculate).
    config_workers:
        Số worker cho config này cụ thể (None = sequential processing).
    """
    import os
    import time
    import logging
    
    # Log process start with PID for debugging
    process_id = os.getpid()
    print(f"[PID {process_id}] Starting worker for config: {config['name']}")
    
    # Monitor GPU in subprocess
    def log_gpu_status(prefix=""):
        try:
            import torch
            
            # Try DirectML first
            try:
                import torch_directml
                if torch_directml.is_available():
                    device_count = torch_directml.device_count()
                    device_name = torch_directml.device_name(0) if device_count > 0 else "Unknown"
                    print(f"[PID {process_id}] {prefix}DirectML: {device_count} device(s) - {device_name}")
                    return
            except ImportError:
                pass
            
            # Check CUDA
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"[PID {process_id}] {prefix}CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            else:
                print(f"[PID {process_id}] {prefix}Using CPU processing")
        except Exception as e:
            print(f"[PID {process_id}] {prefix}GPU monitoring failed: {e}")
    
    log_gpu_status("WORKER START - ")
    
    if embedding_model:
        COMMON_DEFAULTS["embedding_model"] = embedding_model
    
    controller = DatasetController(input_tsv_path, output_base_dir, auto_start_oie=auto_start_oie, silent_mode=True)
    start_time = time.time()
    
    log_gpu_status("AFTER CONTROLLER INIT - ")
        
    success, output_file, final_compliance = controller._run_single_config(config, ranking_workers=ranking_workers, config_workers=config_workers)
    
    log_gpu_status("AFTER CONFIG EXECUTION - ")
        
    end_time = time.time()
    
    print(f"[PID {process_id}] Completed worker for config: {config['name']} in {end_time - start_time:.1f}s")
        
    return {
        "config_name": config["name"], "description": config["description"],
        "success": success, "output_file": output_file or "",
        "execution_time_seconds": end_time - start_time,
        "final_compliance_rate": final_compliance,
        "worker_pid": process_id  # Add PID for debugging
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
# ============================================================
# Main Function: Interactive Dataset Controller Interface
# ============================================================
def data_create_controller_main():
    """Interactive main function for adaptive dataset controller execution.
    
    Provides comprehensive user interface for configuring and executing
    dataset creation with embedding model selection, processing mode options,
    and system optimization recommendations.
    """
    global RUN_CONFIGURATIONS  # Allow runtime configuration override
    print("Integrated Adaptive Dataset Controller for Neural Ranking Models")
    print("="*60)
    
    # Collect input parameters with validation
    input_tsv_path = input("Enter path to input TSV file (query<tab>passage<tab>label): ").strip()
    if not input_tsv_path:
        print("ERROR: Input file path is required for processing!")
        return
    
    output_base_dir = input("Enter output base directory (default: ./training_datasets): ").strip() or "./training_datasets"
    
    # Interactive embedding model selection with optimization recommendations
    print(f"\nCurrent embedding model: {COMMON_DEFAULTS['embedding_model']}")
    change_model = input("Do you want to change the embedding model? (y/n, default: n): ").strip().lower()
    
    if change_model == 'y':
        # Create temporary controller instance to access utility methods
        temp_controller = DatasetController.__new__(DatasetController)
        
        # Show quick presets for common configurations
        temp_controller._utility_quick_model_switch()
        
        # Provide option for detailed model exploration
        advanced = input("\nWant to see all models with detailed specifications? (y/n): ").strip().lower()
        if advanced == 'y':
            temp_controller._utility_display_embedding_models()
            model_choice = input("\nEnter model key manually: ").strip().lower()
            if model_choice and model_choice in AVAILABLE_EMBEDDING_MODELS:
                _utility_set_embedding_model(model_choice)
                model_info = AVAILABLE_EMBEDDING_MODELS[model_choice]
                print(f"Selected: {model_info['model_name']}")
    
    # Display final model configuration for confirmation
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
        print(f"\nFinal embedding model: {final_model} (custom configuration)")
    
    # Processing mode selection with comprehensive options
    mode_choice = input("\nProcessing Mode Selection:\n1. Single-phase (Without OIE only)\n2. Two-phase (Without OIE → With OIE)\n3. Convert existing chunks to OIE\nEnter choice (1-3, default: 1): ").strip()
    
    if mode_choice == '2':
        two_phase = True
        convert_mode = False
    elif mode_choice == '3':
        two_phase = False
        convert_mode = True
    else:
        two_phase = False
        convert_mode = False

    if convert_mode:
        # Existing chunks conversion mode with OpenIE integration
        print("\n=== Convert Existing Chunks to OIE Mode ===")
        source_file = input("Enter path to existing chunks file (TSV): ").strip()
        if not source_file or not os.path.exists(source_file):
            print("ERROR: Source file not found!")
            return
        
        # Initialize OpenIE server for relationship extraction
        try:
            from Tool.OIE import _is_port_open, _start_openie_server
            if not _is_port_open(9000):
                print("Starting OpenIE server for relationship extraction...")
                _start_openie_server(9000)
                
            controller = DatasetController(input_tsv_path, output_base_dir, auto_start_oie=False)
            result_file = controller.convert_existing_chunks_to_oie(source_file)
            
            if result_file:
                print(f"Successfully converted to OIE version: {result_file}")
            else:
                print("Conversion failed")
                
            # Cleanup OpenIE resources
            from Tool.OIE import terminate_openie_processes
            terminate_openie_processes(port=9000, silent=True)
            
        except Exception as e:
            print(f"Error during conversion: {e}")
        
        return

    # Parallel processing configuration with system optimization
    parallel_choice = input("Use parallel processing for chunking phase? (y/n, default: n): ").strip().lower()
    use_parallel = parallel_choice == 'y'
    
    max_workers = None
    ranking_workers = None
    oie_threads = 8  # Default value
    
    if use_parallel:
        print(f"\nWorker Configuration (Max CPU cores: {multiprocessing.cpu_count()})")
        print("=" * 50)
        
        # Chunking/OIE workers
        max_workers_input = input(f"Chunking/OIE workers (default: auto-calculate): ").strip()
        if max_workers_input.isdigit():
            max_workers = min(int(max_workers_input), multiprocessing.cpu_count())
            print(f"Chunking/OIE workers: {max_workers}")
        else:
            print(f"Chunking/OIE workers: auto-calculate")
    
    # Config workers (ask for both parallel and sequential)
    print(f"\nConfig Worker Configuration")
    print("=" * 40)
    print("Workers per config: Each config can use multiple workers for internal batch processing")
    config_workers_input = input(f"Workers per config (default: 1=sequential, recommended: 2-5): ").strip()
    config_workers = None
    if config_workers_input.isdigit():
        config_workers = min(int(config_workers_input), multiprocessing.cpu_count())
        print(f"Workers per config: {config_workers}")
        if use_parallel and max_workers and config_workers:
            total_workers = max_workers * config_workers
            print(f"Total system workers: {total_workers} (configs: {max_workers} × workers per config: {config_workers})")
    else:
        print(f"Workers per config: 1 (sequential)")
    
    # Ranking workers (ask for both parallel and sequential since ranking affects both)
    print(f"\nRanking Worker Configuration")
    print("=" * 40)
    ranking_workers_input = input(f"Ranking workers (default: auto-calculate, recommended: 2-4): ").strip()
    if ranking_workers_input.isdigit():
        ranking_workers = min(int(ranking_workers_input), multiprocessing.cpu_count())
        print(f"Ranking workers: {ranking_workers}")
    else:
        print(f"Ranking workers: auto-calculate")

    # OIE threads (only for two-phase mode)
    if two_phase:
        print(f"\nOIE Thread Configuration (Two-phase mode)")
        print("=" * 45)
        oie_threads_input = input(f"OIE threads (default: 8, recommended: 4-8): ").strip()
        if oie_threads_input.isdigit():
            oie_threads = min(int(oie_threads_input), multiprocessing.cpu_count())
            print(f"OIE threads: {oie_threads}")
        else:
            print(f"OIE threads: 8 (default)")

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
    
    if use_parallel:
        chunking_info = f"{max_workers or 'auto'} chunking workers"
        ranking_info = f"{ranking_workers or 'auto'} ranking workers"
        config_info = f"{config_workers or 1} workers per config"
        if two_phase:
            oie_info = f"{oie_threads} OIE threads"
            print(f"Parallel mode: {chunking_info}, {config_info}, {ranking_info}, {oie_info}")
        else:
            print(f"Parallel mode: {chunking_info}, {config_info}, {ranking_info}")
    else:
        ranking_info = f"{ranking_workers or 'auto'} ranking workers"
        config_info = f"{config_workers or 1} workers per config"
        if two_phase:
            oie_info = f"{oie_threads} OIE threads"
            print(f"Sequential mode: {config_info}, {ranking_info}, {oie_info}")
        else:
            print(f"Sequential mode: {config_info}, {ranking_info}")
    
    print("Starting process...")
    
    try:
        # In two-phase mode, don't start OIE server initially - it will be started in phase 2
        controller = DatasetController(input_tsv_path, output_base_dir, auto_start_oie=False)

        if two_phase:
            controller.process_create_all_datasets_two_phase(
                non_oie_parallel=use_parallel, 
                max_workers=max_workers, 
                ranking_workers=ranking_workers,
                oie_threads=oie_threads
            )
        else:
            controller.data_create_controller_main(
                use_parallel=use_parallel, 
                max_workers=max_workers, 
                ranking_workers=ranking_workers,
                config_workers=config_workers
            )
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"A critical error occurred in main: {e}")
        traceback.print_exc()

# -------------------------------------------------------------------
# ============================================================
# Worker Functions: Parallel batch processing with dedicated models
# ============================================================
def helper_process_single_batch(batch_df: pd.DataFrame, config: dict, batch_num: int, 
                               embedding_model: str = None, worker_id: int = None) -> List[List[Any]]:
    """Process single batch in dedicated worker process with isolated embedding model.
    
    Executes chunking operations for a single data batch in a separate process,
    ensuring memory isolation and dedicated model instances for optimal performance.
    
    Args:
        batch_df (pd.DataFrame): DataFrame containing batch data to process
        config (dict): Configuration dictionary containing chunking parameters and method selection
        batch_num (int): Batch number identifier for logging and debugging
        embedding_model (str, optional): Specific embedding model for this worker process
        worker_id (int, optional): Unique worker identifier for process tracking
        
    Returns:
        List[List[Any]]: Processed chunked rows ready for dataset creation
        
    Note:
        Each worker loads its own embedding model instance to prevent memory conflicts
        and ensure stable parallel processing across multiple cores.
    """
    import os
    import gc
    from pathlib import Path
    
    # Identify process for debugging and resource tracking
    process_id = os.getpid()
    worker_tag = f"W{worker_id}" if worker_id is not None else "UNK"
    
    try:
        print(f"[PID {process_id}|{worker_tag}] Processing batch {batch_num} with {len(batch_df)} rows")
        
        # Initialize dedicated embedding model for this worker process
        if embedding_model:
            print(f"[PID {process_id}|{worker_tag}] Loading dedicated embedding model: {embedding_model}")
            # Configure model for this worker process
            COMMON_DEFAULTS["embedding_model"] = embedding_model
        
        # Assess GPU/CPU availability for this worker with DirectML support
        try:
            import torch
            
            # Check DirectML availability first (AMD GPU support)
            try:
                import torch_directml
                if torch_directml.is_available():
                    device_count = torch_directml.device_count()
                    device_name = torch_directml.device_name(0) if device_count > 0 else "Unknown"
                    print(f"[PID {process_id}|{worker_tag}] Using DirectML: {device_count} device(s) - {device_name}")
                else:
                    print(f"[PID {process_id}|{worker_tag}] DirectML not available")
            except ImportError:
                print(f"[PID {process_id}|{worker_tag}] DirectML not installed")
            
            # Check CUDA as fallback
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"[PID {process_id}|{worker_tag}] CUDA available: {gpu_memory:.2f}GB allocated")
            else:
                print(f"[PID {process_id}|{worker_tag}] Using CPU processing")
        except ImportError:
            print(f"[PID {process_id}|{worker_tag}] PyTorch not available, using CPU processing")
        
        # Execute batch processing using dedicated chunking logic
        output_rows = []
        method_choice = config["method_choice"]
        method_params = config["params"].copy()

        # Configure OpenIE integration based on availability and settings
        use_oie = config.get("include_oie", False) and OIE_AVAILABLE_GLOBALLY
        method_params['include_oie'] = use_oie
        method_params['save_raw_oie'] = use_oie and config.get("save_raw_oie", False)

        # Map method choices to their corresponding chunking functions
        chunking_function_map = {
            "1": semantic_chunk_passage_from_grouping_logic,      # Semantic grouping
            "2": chunk_passage_text_splitter,                    # Semantic splitting  
            "3": chunk_passage_text_splitter,                    # Text splitting
        }
        chunking_function = chunking_function_map.get(method_choice)

        if not chunking_function:
            print(f"[PID {process_id}|{worker_tag}] No valid chunking function for method_choice '{method_choice}'. Skipping batch.")
            return []

        # Initialize memory monitoring for resource optimization
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # Convert to MB

        for row_idx, row_data in batch_df.iterrows():
            try:
                query_text, passage_text, label = row_data.iloc[0], row_data.iloc[1], row_data.iloc[2]
                query_id = f"Q{row_idx}"
                passage_id = f"P{row_idx}"

                if not isinstance(passage_text, str) or not passage_text.strip():
                    continue

                # Prepare parameters for the chunking function
                common_params = {
                    'passage_text': passage_text,
                    'doc_id': f"{query_id}_{passage_id}",
                    'output_dir': None,  # Not needed for batch processing
                    'silent': True,
                    **method_params
                }
                # Add semantic-specific parameters for methods 1 and 2
                if method_choice in ["1", "2"]:  # Semantic grouping and semantic splitting
                    common_params['embedding_model'] = COMMON_DEFAULTS["embedding_model"]
                    common_params['device'] = COMMON_DEFAULTS["device_preference"]

                # Execute the selected chunking algorithm
                chunked_tuples: List[Tuple[str, str, Optional[str]]] = chunking_function(**common_params)

                if not chunked_tuples:
                    continue
                
                # Format output rows for dataset creation
                for gen_chunk_id, chunk_text, oie_string in chunked_tuples:
                    output_rows.append([
                        query_id, query_text, passage_id, passage_text, label,
                        gen_chunk_id, chunk_text, oie_string if oie_string else ""
                    ])
                    
                # Immediately free memory from processed chunks
                del chunked_tuples
                
            except Exception as e:
                print(f"[PID {process_id}|{worker_tag}] Error processing row {row_idx}: {e}")

        # Final memory cleanup and performance statistics
        gc.collect()
        final_memory = process.memory_info().rss / (1024**2)  # Convert to MB
        
        print(f"[PID {process_id}|{worker_tag}] Batch {batch_num} completed: {len(output_rows)} chunks generated")
        print(f"[PID {process_id}|{worker_tag}] Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB")
        
        return output_rows
        
    except Exception as e:
        print(f"[PID {process_id}|{worker_tag}] Batch {batch_num} failed with error: {e}")
        return []


# ============================================================
# Main Entry Point: Dataset Controller Execution
# ============================================================
if __name__ == "__main__":
    """Main entry point for dataset controller execution with performance tracking."""
    start_time = time.time()
    
    data_create_controller_main()
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")