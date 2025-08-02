# -*- coding: utf-8 -*-
from __future__ import annotations

# ---------------------------------------------------------------------------
# Suppress noisy FutureWarnings & verbose library logs
# ---------------------------------------------------------------------------
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

# ==== Default constants ====
BATCH_SIZE = 600  # dòng/đợt đọc pandas
COMMON_DEFAULTS = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "device_preference": "dml",  # "cuda", "dml", hoặc "cpu"
}

# ==== Worker initializer =====================================================

def _worker_init(model_name: str, device_pref: str):
    """Nạp SentenceTransformer + spaCy 1 lần cho mỗi process."""
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers import SentenceTransformer
    import spacy

    global _GLOBALS  # type: ignore
    _GLOBALS = {}

    # spaCy sentencizer tối giản
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
) -> List[List[Any]]:
    """Chunk các dòng trong batch và trả về list row cho TSV output."""
    results: List[List[Any]] = []

    global _GLOBALS  # type: ignore
    nlp = _GLOBALS.get("nlp")
    model = _GLOBALS.get("model")
    if model is None:
        raise RuntimeError("_worker_init chưa chạy – model None")

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

        if not isinstance(passage, str) or not passage.strip():
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
):
    """Chạy chunking cho 1 cấu hình."""
    import time

    # Use provided models or fall back to defaults
    actual_embedding_model = embedding_model or COMMON_DEFAULTS["embedding_model"]
    actual_device_preference = device_preference or COMMON_DEFAULTS["device_preference"]

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
                exe.submit(_process_batch, df, config, idx, actual_embedding_model, actual_device_preference): idx 
                for idx, df in all_batches
            }
            batch_results = []
            for fut in as_completed(futures):
                batch_results.extend(fut.result())
    else:
        batch_results = []
        for idx, df in all_batches:
            batch_results.extend(_process_batch(df, config, idx, actual_embedding_model, actual_device_preference))

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
        "description": "Pure semantic chunking without size limits"
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
        "description": "Semantic grouping with threshold ranges - no size constraints"
    },
]

# ============================================================
# INTERACTIVE TEXT-UI (bắt chước data_create_controller.py)
# ============================================================

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
    print(f"\nCurrent embedding model: {COMMON_DEFAULTS['embedding_model']}")
    if input("Change model? (y/N): ").strip().lower() == "y":
        print("\nQUICK MODEL PRESETS:")
        for key, (model_name, desc) in _MODEL_PRESETS.items():
            print(f"  {key}. {model_name} - {desc}")
        choice = input("Select preset (1-5) or press ENTER to keep current: ").strip()
        if choice in _MODEL_PRESETS:
            selected = _MODEL_PRESETS[choice][0]
            if selected == "auto":
                selected = _recommend_auto_model()
            COMMON_DEFAULTS["embedding_model"] = selected
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
    print(f"Embedding    : {COMMON_DEFAULTS['embedding_model']}")
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

    # Chạy tất cả configs được chọn
    for i, cfg in enumerate(selected_cfgs, 1):
        print(f"\n[CONFIG {i}/{len(selected_cfgs)}] Starting {cfg['name']}...")
        run_config(
            cfg, 
            input_path, 
            out_dir, 
            cfg_workers,
            embedding_model=COMMON_DEFAULTS["embedding_model"],
            device_preference=COMMON_DEFAULTS["device_preference"]
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
    ap.add_argument(
        "--configs",
        help="Comma separated config names or numbers to run (default all)",
    )
    args = ap.parse_args()

    COMMON_DEFAULTS["embedding_model"] = args.embedding_model
    COMMON_DEFAULTS["device_preference"] = args.device.lower()

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

    # pool cho configs
    if args.workers > 1:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(COMMON_DEFAULTS["embedding_model"], COMMON_DEFAULTS["device_preference"]),
        ) as exe:
            futs = {
                exe.submit(
                    run_config, 
                    cfg, 
                    input_path, 
                    out_dir, 
                    args.config_workers,
                    COMMON_DEFAULTS["embedding_model"],
                    COMMON_DEFAULTS["device_preference"]
                ): cfg["name"]
                for cfg in configs
            }
            for fut in as_completed(futs):
                fut.result()
    else:
        for cfg in configs:
            run_config(
                cfg, 
                input_path, 
                out_dir, 
                args.config_workers,
                embedding_model=COMMON_DEFAULTS["embedding_model"],
                device_preference=COMMON_DEFAULTS["device_preference"]
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