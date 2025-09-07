"""Common utilities for semantic splitting and grouping.

Provides:
 - normalize_device
 - estimate_optimal_batch_size / optimize_gpu_batch_size
 - embed_sentences_batched
 - create_similarity_matrix (GPU accelerated when available)
 - format_oie_triples_to_string
 - extract_oie_for_chunk
 - save_raw_oie_data

All functions keep signature simplicity and silent flag to suppress logs.
"""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
import json
import hashlib
from datetime import datetime
from pathlib import Path
import logging

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # fallback

from Tool.Sentence_Embedding import sentence_embedding as embed_text_list

# Optional TPU (PyTorch/XLA)
try:  # pragma: no cover
    import os as _os
    import importlib as _importlib
    _torch_xla = _importlib.import_module('torch_xla.core.xla_model')
except Exception:
    _torch_xla = None
from Tool.OIE import extract_relations_from_paragraph

# ---------------- Device & Batch Utilities ---------------- #

def normalize_device(device: Optional[object]) -> str:
    try:
        if torch is not None and torch.cuda.is_available():
            cuda_available = True
        else:
            cuda_available = False
    except Exception:
        cuda_available = False
    device_str = str(device).lower() if device is not None else None
    # Auto-detect preference: CUDA > TPU(XLA) > CPU
    if device_str in (None, "auto"):
        if cuda_available:
            return "cuda"
        if _torch_xla is not None:
            return "tpu"
        return "cpu"
    if device_str.startswith("cuda"):
        return "cuda" if cuda_available else "cpu"
    if device_str in ("tpu", "xla"):
        return "tpu" if _torch_xla is not None else ("cuda" if cuda_available else "cpu")
    return "cpu"

def estimate_optimal_batch_size(sentences: List[str], base_batch_size: int, device: str) -> int:
    if device != "cuda":
        if len(sentences) >= 50:
            return min(base_batch_size, 16)
        return min(base_batch_size, max(8, len(sentences)))
    if not sentences:
        return base_batch_size
    avg_len = sum(len(s) for s in sentences) / max(1, len(sentences))
    if avg_len < 50:
        return min(256, max(base_batch_size, 64), len(sentences))
    if avg_len < 100:
        return min(128, max(base_batch_size, 32), len(sentences))
    if avg_len < 200:
        return min(64, max(base_batch_size, 16), len(sentences))
    return min(32, max(base_batch_size, 8), len(sentences))

# Legacy alias for grouping code
optimize_gpu_batch_size = estimate_optimal_batch_size

# ---------------- Embedding ---------------- #

def embed_sentences_batched(
    sentences: List[str],
    model_name: str,
    base_batch_size: int = 32,
    device: Optional[object] = None,
    silent: bool = True,
) -> np.ndarray:
    if not sentences:
        return np.array([])
    device_pref = normalize_device(device)
    batch_size = estimate_optimal_batch_size(sentences, base_batch_size, device_pref)
    all_embeddings: List[np.ndarray] = []
    total = len(sentences)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = sentences[start:end]
        cur_bs = len(batch) if device_pref == "cuda" else min(len(batch), 16)
        idx = 0
        sub_embeds: List[np.ndarray] = []
        while idx < len(batch):
            sub_end = min(idx + cur_bs, len(batch))
            sub_batch = batch[idx:sub_end]
            try:
                embeds = embed_text_list(
                    sub_batch,
                    model_name=model_name,
                    batch_size=max(1, cur_bs),
                    device_preference=device_pref,
                )
                if embeds is not None and len(embeds) > 0:
                    sub_embeds.append(np.asarray(embeds))
                idx = sub_end
            except Exception as e:  # pragma: no cover - defensive
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
                if is_oom and device_pref == "cuda" and cur_bs > 1:
                    cur_bs = max(1, cur_bs // 2)
                    if torch is not None and torch.cuda.is_available():
                        try: torch.cuda.empty_cache()  # noqa: E701
                        except Exception: pass
                    if not silent:
                        print(f"[embed] OOM reduce sub-batch -> {cur_bs}")
                    continue
                if not silent:
                    print(f"[embed] error batch {start}:{sub_end}: {e}")
                idx = sub_end
        if sub_embeds:
            try:
                all_embeddings.append(np.vstack(sub_embeds))
            except Exception:
                all_embeddings.append(np.asarray([v for arr in sub_embeds for v in arr]))
    if not all_embeddings:
        return np.array([])
    try:
        return np.vstack(all_embeddings)
    except Exception:
        return np.asarray([v for arr in all_embeddings for v in arr])

# ---------------- Similarity Matrix ---------------- #

def create_similarity_matrix(
    sentences: List[str],
    model_name: str,
    batch_size: int = 32,
    device: Optional[str] = "cuda",
    silent: bool = True,
) -> Optional[np.ndarray]:
    if len(sentences) < 2:
        return None
    device_str = normalize_device(device)
    embs = embed_sentences_batched(sentences, model_name, base_batch_size=batch_size, device=device_str, silent=silent)
    if embs is None or embs.size == 0 or embs.shape[0] != len(sentences):
        return None
    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs = embs / norms
    try:
        if device_str == "cuda" and torch is not None and torch.cuda.is_available():
            t = torch.tensor(embs, dtype=torch.float32, device="cuda")
            sim = torch.mm(t, t.t()).cpu().numpy()
            del t
            if torch.cuda.is_available():
                try: torch.cuda.empty_cache()
                except Exception: pass
            return sim
        elif device_str == "tpu" and _torch_xla is not None and torch is not None:
            try:
                xdev = _torch_xla.xla_device()
                t = torch.tensor(embs, dtype=torch.float32, device=xdev)
                sim_x = torch.mm(t, t.t())
                # XLA needs mark_step to ensure computation materializes
                try:
                    import torch_xla.core.xla_model as xm  # type: ignore
                    xm.mark_step()
                except Exception:
                    pass
                sim = sim_x.cpu().numpy()
                del t, sim_x
                return sim
            except Exception:
                # Fallback to CPU if XLA path fails
                return embs @ embs.T
        else:
            # CPU fallback (manual cosine using dot since normalized)
            return embs @ embs.T
    except Exception:
        return embs @ embs.T

# ---------------- OIE Utilities ---------------- #

def format_oie_triples_to_string(triples_list: List[Dict[str, str]], max_triples: Optional[int] = None) -> Optional[str]:
    if not triples_list:
        return None
    formatted: List[str] = []
    triples_to_process = triples_list[:max_triples] if max_triples is not None else triples_list
    for triple in triples_to_process:
        s = str(triple.get('subject','')).replace('\t',' ').replace('\n',' ').strip()
        r = str(triple.get('relation','')).replace('\t',' ').replace('\n',' ').strip()
        o = str(triple.get('object','')).replace('\t',' ').replace('\n',' ').strip()
        if s and r and o:
            formatted.append(f"{s} {r} {o}.")
    if not formatted:
        return None
    return " ".join(formatted).strip()

def extract_oie_for_chunk(chunk_text: str, max_triples: Optional[int] = None, silent: bool = True) -> Optional[str]:
    if not chunk_text or not chunk_text.strip():
        return None
    try:
        relations = extract_relations_from_paragraph(chunk_text)
        if relations:
            return format_oie_triples_to_string(relations, max_triples=max_triples)
        return None
    except Exception as e:  # pragma: no cover
        if not silent:
            print(f"[oie] error: {e}")
        return None

def save_raw_oie_data(oie_data: List[Dict], doc_or_chunk_id: str, output_dir: str, method_name: str) -> Optional[str]:
    try:
        out_dir = Path(output_dir) / "raw_oie_data"
        out_dir.mkdir(exist_ok=True)
        filename = f"{method_name}_raw_oie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = out_dir / filename
        entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
            "id": doc_or_chunk_id,
            "raw_oie_relations": oie_data,
            "total_relations": sum(e.get('relation_count', 0) for e in oie_data)
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                existing = []
            existing.append(entry)
        else:
            existing = [entry]
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding='utf-8')
        return str(path)
    except Exception as e:  # pragma: no cover
        print(f"Warning: cannot save raw OIE data: {e}")
        return None

def analyze_similarity_distribution(sim_matrix):
    if not isinstance(sim_matrix, np.ndarray) or sim_matrix.ndim != 2 or sim_matrix.shape[0] < 2:
        return None
    upper = np.triu_indices_from(sim_matrix, k=1)
    sims = sim_matrix[upper]
    eps = 1e-5
    filtered = sims[sims < (1.0 - eps)]
    if filtered.size == 0:
        if sims.size > 0:
            mx = float(np.max(sims))
            return {k: mx for k in ['min','max','mean','std','p10','p25','p50','p75','p80','p85','p90','p95']}
        return None
    pct = {f'p{p}': float(np.percentile(filtered, p)) for p in [10,25,50,75,80,85,90,95]}
    stats = {
        'min': float(np.min(filtered)),
        'max': float(np.max(filtered)),
        'mean': float(np.mean(filtered)),
        'std': float(np.std(filtered)),
    }
    stats.update(pct)
    return stats

__all__ = [
    "normalize_device",
    "estimate_optimal_batch_size",
    "optimize_gpu_batch_size",
    "embed_sentences_batched",
    "create_similarity_matrix",
    "format_oie_triples_to_string",
    "extract_oie_for_chunk",
    "save_raw_oie_data",
    "analyze_similarity_distribution",
    "init_logger",
    "log_msg",
]

# ---------------- Logging Utilities ---------------- #

_LOGGER_INITIALIZED = False

def init_logger(name: str = "semantic", level: int = logging.INFO) -> logging.Logger:
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)
    if not _LOGGER_INITIALIZED or not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s', '%H:%M:%S')
        handler.setFormatter(formatter)
        # Avoid duplicate handlers
        if not logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False
        _LOGGER_INITIALIZED = True
    return logger

_LEVEL_MAP = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'error': logging.ERROR,
}

def log_msg(silent: bool, msg: str, level: str = 'info', component: str = None):
    if silent:
        return
    base = 'semantic' if component is None else f'semantic.{component}'
    logger = init_logger(base)
    lvl = _LEVEL_MAP.get(level.lower(), logging.INFO)
    logger.log(lvl, msg)
