#!/usr/bin/env python3
import os
import logging
import psutil
import subprocess
import socket
import time
import shutil
from pathlib import Path
from typing import List, Dict, Optional

try:
    from pyopenie import OpenIE5  # type: ignore
except Exception as e:  # pragma: no cover
    OpenIE5 = None  # type: ignore
    logging.warning("pyopenie not available: %s", e)

"""CLI (Ubuntu-friendly):
Examples:
  python3 Tool/OIE_ubuntu.py input.tsv output.tsv --batch-size 50 --port 9001
  OPENIE_JAR_PATH=/path/to/openie-assembly-5.0-SNAPSHOT.jar python3 Tool/OIE_ubuntu.py in.tsv out.tsv
  OPENIE_XMS_GB=2 OPENIE_XMX_GB=6 python3 Tool/OIE_ubuntu.py in.tsv out.tsv
"""

# -------------------- Singleton OpenIE5 client ---------------------------
_GLOBAL_CLIENT: Optional[OpenIE5] = None
_SERVER_LAUNCHED = False


# -------------------- Utilities ---------------------------
def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("localhost", port)) == 0


def _which_java() -> Optional[str]:
    return shutil.which("java")


def _resolve_jar_path() -> Optional[Path]:
    # 1) ENV override
    env_path = os.environ.get("OPENIE_JAR_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None
    # 2) Project default
    project_root = Path(__file__).resolve().parent.parent
    jar = project_root / "OpenIE-standalone" / "openie-assembly-5.0-SNAPSHOT.jar"
    return jar if jar.exists() else None


def _compute_java_memory_flags() -> List[str]:
    # Fixed per requirement: -Xms8g -Xmx16g -XX:+UseG1GC
    return ["-Xms8g", "-Xmx16g", "-XX:+UseG1GC"]


def _kill_processes_on_port(port: int, silent: bool = False) -> None:
    """Kill any process currently listening on the given TCP port."""
    try:
        conns = psutil.net_connections(kind='inet')
    except Exception:
        conns = []
    pids = set()
    for c in conns:
        try:
            if c.laddr and hasattr(c.laddr, 'port') and c.laddr.port == port and c.status == psutil.CONN_LISTEN:
                if c.pid:
                    pids.add(c.pid)
        except Exception:
            continue
    for pid in pids:
        try:
            p = psutil.Process(pid)
            if not silent:
                logging.warning("[OIE] Terminating process %d on port %d", pid, port)
            p.terminate()
            try:
                p.wait(timeout=3)
            except psutil.TimeoutExpired:
                if not silent:
                    logging.warning("[OIE] Force killing process %d", pid)
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


# -----------------------------------------------------------------
# Start OpenIE5 server once per process (idempotent) - Ubuntu friendly
# -----------------------------------------------------------------
def _start_openie_server(port: int) -> None:
    global _SERVER_LAUNCHED
    if _SERVER_LAUNCHED:
        logging.warning("[OIE] Server launch already attempted (port %d).", port)
        return

    # If port is in use, kill the occupying process as requested
    if _is_port_open(port):
        logging.warning("[OIE] Port %d is occupied. Attempting to terminate the process...", port)
        _kill_processes_on_port(port)
        time.sleep(0.5)

    java_path = _which_java()
    jar_path = _resolve_jar_path()
    if not java_path:
        logging.warning("[OIE] 'java' not found. Please install OpenJDK: sudo apt-get install -y default-jre")
        return
    if not jar_path:
        logging.warning("[OIE] OpenIE JAR not found. Set OPENIE_JAR_PATH or place jar in OpenIE-standalone/.")
        return

    mem_flags = _compute_java_memory_flags()
    logging.warning("[OIE] Launching OpenIE5 server on port %d with %s", port, " ".join(mem_flags))
    print(f"[OIE] Launching OpenIE5 server on port {port} with {' '.join(mem_flags)}")

    cmd = [
        java_path,
        "-server",
        *mem_flags,
        "-jar", str(jar_path),
        "--httpPort", str(port),
    ]

    try:
        subprocess.Popen(
            cmd,
            cwd=str(jar_path.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach on Linux/Ubuntu
        )
        _SERVER_LAUNCHED = True
        # Non-blocking launch: return immediately without waiting for readiness
        logging.warning("[OIE] OpenIE5 server launch triggered (non-blocking) on port %d.", port)
        print(f"[OIE] Server launch triggered (non-blocking) on port {port}.")
        return
    except Exception as exc:
        logging.warning("[OIE] Failed to launch OpenIE5: %s", exc)


def _get_global_client(port: int = 9000) -> OpenIE5:
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        print(f"[OIE] Ensuring server on port {port}...")
        if not _is_port_open(port):
            print(f"[OIE] Port {port} not open. Attempting to start server...")
            _start_openie_server(port)
        else:
            print(f"[OIE] Port {port} is already open.")
        if OpenIE5 is None:  # pragma: no cover
            raise RuntimeError("pyopenie is not available. pip install pyopenie")
        _GLOBAL_CLIENT = OpenIE5(f"http://localhost:{port}")
    return _GLOBAL_CLIENT


def _convert_openie5(raw_results: List[Dict]) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    for item in raw_results:
        extraction = item.get("extraction", {})
        arg1 = extraction.get("arg1", {}).get("text", "").strip()
        rel = extraction.get("rel", {}).get("text", "").strip()
        if not arg1 or not rel:
            continue
        for arg2 in extraction.get("arg2s", []):
            obj = arg2.get("text", "").strip()
            if obj:
                triples.append({
                    "subject": arg1,
                    "relation": rel,
                    "object": obj,
                })
    return triples


DEFAULT_PROPERTIES = {
    "annotators": "tokenize,ssplit,pos,lemma,depparse,openie",
    "outputFormat": "json",
    "timeout": "600000",
    "openie.triple.strict": "false",
    "openie.max_entailments_per_clause": "1000",
    "openie.affinity_probability_cap": "0.1",
}


def extract_relations_from_paragraph(
    paragraph_text: str,
    port: int = 9000,
    silent: bool = True
) -> List[Dict[str, str]]:
    if not paragraph_text.strip():
        return []

    results: List[Dict[str, str]] = []
    original_levels = {}
    if silent:
        import logging as _logging
        for logger_name in ['stanford', 'corenlp', 'stanza', 'java']:
            logger = _logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(_logging.CRITICAL)

    try:
        # Fast-fail if port is not yet open to avoid hanging on first batches
        if not _is_port_open(port):
            print(f"[OIE] Port {port} is not open yet. Skipping extraction for this row.")
            return []

        client = _get_global_client(port=port)

        # Enforce per-call timeout using a worker thread to avoid hard hangs
        import threading
        raw_results_holder: Dict[str, Optional[List[Dict]]] = {"data": None}
        error_holder: Dict[str, Optional[Exception]] = {"err": None}

        def _worker():
            try:
                raw = client.extract(paragraph_text)
                raw_results_holder["data"] = raw
            except Exception as _e:
                error_holder["err"] = _e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(8.0)  # 8s timeout

        if t.is_alive():
            # Timed out: skip this row to keep pipeline moving
            print("[OIE] extract() timeout, skipping this row.")
            return []

        if error_holder["err"] is not None:
            raise error_holder["err"]

        raw_results = raw_results_holder["data"] or []
        results.extend(_convert_openie5(raw_results))
    except Exception as e:
        if not silent:
            print(f"Error in OIE extraction: {e}")
    finally:
        if silent and original_levels:
            import logging as _logging
            for logger_name, original_level in original_levels.items():
                logger = _logging.getLogger(logger_name)
                logger.setLevel(original_level)

    # Deduplicate
    unique: List[Dict[str, str]] = []
    seen = set()
    for r in results:
        key = f"{r['subject'].strip().lower()}|{r['relation'].strip().lower()}|{r['object'].strip().lower()}"
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def format_oie_triples_as_text(triples: List[Dict[str, str]]) -> str:
    if not triples:
        return ""
    parts = []
    for t in triples:
        s = t.get("subject", "").strip()
        r = t.get("relation", "").strip()
        o = t.get("object", "").strip()
        if s and r and o:
            parts.append(f"({s}; {r}; {o})")
    return " ".join(parts)


def process_chunk_tsv_with_oie(
    input_tsv_path: str,
    output_tsv_path: str,
    chunk_text_column: str = "chunk_text",
    port: int = 9000,
    silent: bool = True,
    batch_size: int = 100
) -> None:
    import pandas as pd
    import csv
    from tqdm import tqdm

    if not silent:
        print(f"Processing TSV file: {input_tsv_path}")
        print(f"Output will be saved to: {output_tsv_path}")

    try:
        df = pd.read_csv(input_tsv_path, sep='\t', dtype=str)
    except Exception as e:
        raise ValueError(f"Error reading input TSV file: {e}")

    required_columns = ["query_id", "document_id", chunk_text_column, "label"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["raw_oie_data"] = ""
    df["raw_oie_data_plus_chunk_text"] = ""

    total_batches = (len(df) + batch_size - 1) // batch_size
    if not silent:
        print(f"Processing in {total_batches} batches of {batch_size} rows each...")

    for b in tqdm(range(total_batches), desc="Processing batches", disable=silent):
        s_idx = b * batch_size
        e_idx = min((b + 1) * batch_size, len(df))
        for idx in range(s_idx, e_idx):
            chunk_text = df.loc[idx, chunk_text_column]
            if pd.isna(chunk_text) or not str(chunk_text).strip():
                df.loc[idx, "raw_oie_data"] = ""
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = str(chunk_text) if not pd.isna(chunk_text) else ""
                continue
            try:
                triples = extract_relations_from_paragraph(str(chunk_text), port=port, silent=True)
                oie_text = format_oie_triples_as_text(triples)
                df.loc[idx, "raw_oie_data"] = oie_text
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = f"{chunk_text} {oie_text}".strip()
            except Exception as e:
                if not silent:
                    print(f"Error processing row {idx}: {e}")
                df.loc[idx, "raw_oie_data"] = ""
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = str(chunk_text)

    try:
        df.to_csv(output_tsv_path, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
        if not silent:
            print(f"Successfully saved output to: {output_tsv_path}")
    except Exception as e:
        raise ValueError(f"Error saving output TSV file: {e}")


def process_chunk_tsv_with_oie_json_format(
    input_tsv_path: str,
    output_tsv_path: str,
    chunk_text_column: str = "chunk_text",
    port: int = 9000,
    silent: bool = True,
    batch_size: int = 100
) -> None:
    import pandas as pd
    import csv
    import json
    from tqdm import tqdm

    try:
        df = pd.read_csv(input_tsv_path, sep='\t', dtype=str)
    except Exception as e:
        raise ValueError(f"Error reading input TSV file: {e}")

    required_columns = ["query_id", "document_id", chunk_text_column, "label"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["raw_oie_data_json"] = ""
    df["raw_oie_data_plus_chunk_text"] = ""

    total_batches = (len(df) + batch_size - 1) // batch_size
    for b in tqdm(range(total_batches), desc="Processing batches", disable=silent):
        s_idx = b * batch_size
        e_idx = min((b + 1) * batch_size, len(df))
        for idx in range(s_idx, e_idx):
            chunk_text = df.loc[idx, chunk_text_column]
            if pd.isna(chunk_text) or not str(chunk_text).strip():
                df.loc[idx, "raw_oie_data_json"] = "[]"
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = str(chunk_text) if not pd.isna(chunk_text) else ""
                continue
            try:
                triples = extract_relations_from_paragraph(str(chunk_text), port=port, silent=True)
                df.loc[idx, "raw_oie_data_json"] = json.dumps(triples, ensure_ascii=False)
                oie_text = format_oie_triples_as_text(triples)
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = f"{chunk_text} {oie_text}".strip()
            except Exception as e:
                if not silent:
                    print(f"Error processing row {idx}: {e}")
                df.loc[idx, "raw_oie_data_json"] = "[]"
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = str(chunk_text)

    try:
        df.to_csv(output_tsv_path, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
        if not silent:
            print(f"Successfully saved JSON output to: {output_tsv_path}")
    except Exception as e:
        raise ValueError(f"Error saving output TSV file: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ubuntu-friendly OpenIE TSV processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 Tool/OIE_ubuntu.py input.tsv output.tsv
  python3 Tool/OIE_ubuntu.py input.tsv output.tsv --json-format --port 9001
  OPENIE_JAR_PATH=/path/to/openie.jar python3 Tool/OIE_ubuntu.py input.tsv output.tsv
        """
    )
    parser.add_argument("input_tsv", help="Input TSV file path")
    parser.add_argument("output_tsv", help="Output TSV file path")
    parser.add_argument("--chunk-column", default="chunk_text")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--json-format", action="store_true")
    parser.add_argument("--silent", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input_tsv)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_tsv}' does not exist.")
        return 1

    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.json_format:
            process_chunk_tsv_with_oie_json_format(
                input_tsv_path=args.input_tsv,
                output_tsv_path=args.output_tsv,
                chunk_text_column=args.chunk_column,
                port=args.port,
                silent=args.silent,
                batch_size=args.batch_size,
            )
        else:
            process_chunk_tsv_with_oie(
                input_tsv_path=args.input_tsv,
                output_tsv_path=args.output_tsv,
                chunk_text_column=args.chunk_column,
                port=args.port,
                silent=args.silent,
                batch_size=args.batch_size,
            )
        if not args.silent:
            print("Processing completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


