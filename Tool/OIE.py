import logging
import psutil
import subprocess
import socket
import time
from pathlib import Path
from typing import List, Dict, Optional
from pyopenie import OpenIE5  # type: ignore

"""CLI:
# Cơ bản
python Tool/OIE.py input_chunks.tsv output_with_oie.tsv

# Với các tùy chọn
python Tool/OIE.py input_chunks.tsv output_with_oie.tsv --batch-size 50 --port 9001

# Với JSON format
python Tool/OIE.py input_chunks.tsv output_with_oie.tsv --json-format
"""

# -------------------- Singleton OpenIE5 client ---------------------------
_GLOBAL_CLIENT: Optional[OpenIE5] = None
# Track whether we already attempted to launch server in this process
_SERVER_LAUNCHED = False


# -------------------- Kiểm tra & khởi động server ------------------------


def _is_port_open(port: int) -> bool:
    """Trả về True nếu localhost:port đang lắng nghe."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("localhost", port)) == 0


# -----------------------------------------------------------------
# Start OpenIE5 server once per process (idempotent)
# -----------------------------------------------------------------
def _start_openie_server(port: int) -> None:
    """Thử khởi động OpenIE5 server (nền) nếu chưa chạy."""
    global _SERVER_LAUNCHED

    # Nếu đã launch trong process hoặc port đã mở → bỏ qua
    if _SERVER_LAUNCHED or _is_port_open(port):
        logging.warning("[OIE] Server đã chạy hoặc đã được khởi động trong tiến trình (cổng %d).", port)
        return

    # Mặc định thư mục jar nằm song song root dự án
    project_root = Path(__file__).resolve().parent.parent
    jar_dir = project_root / "OpenIE-standalone"
    jar_path = jar_dir / "openie-assembly-5.0-SNAPSHOT.jar"

    if not jar_path.exists():
        logging.warning("[OIE] Không tìm thấy %s, bỏ qua việc tự khởi động OpenIE5.", jar_path)
        return

    logging.warning("[OIE] Đang khởi chạy OpenIE5 server (cổng %d)...", port)

    cmd = [
        "java",
        "-server",
        "-Xms10g",          # heap khởi đầu 8 GB
        "-Xmx16g",         # heap tối đa 12 GB  (đủ mà dư 8 GB cho Python)
        "-XX:+UseG1GC",    # GC G1 – ổn định trên heap lớn
        "-jar", str(jar_path),
        "--httpPort", str(port),
    ]

    try:
        subprocess.Popen(cmd, cwd=str(jar_dir), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _SERVER_LAUNCHED = True  # Đánh dấu đã launch
        # Đợi server lên tối đa 30 giây
        for _ in range(30):
            if _is_port_open(port):
                logging.warning("[OIE] Đã khởi chạy OpenIE5 server (cổng %d).", port)
                return
            time.sleep(1)
        logging.warning("[OIE] Khởi động OpenIE5 timeout – cổng %d vẫn chưa mở.", port)
    except Exception as exc:
        logging.warning("[OIE] Không thể khởi động OpenIE5: %s", exc)


def _get_global_client(port: int = 9000) -> OpenIE5:
    """Trả về một client OpenIE5 duy nhất (kết nối lại nếu đã có)."""
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        # Nếu server chưa chạy, cố gắng tự khởi động
        if not _is_port_open(port):
            _start_openie_server(port)

        # Khởi tạo kết nối tới server OpenIE5
        _GLOBAL_CLIENT = OpenIE5(f"http://localhost:{port}")
    return _GLOBAL_CLIENT


# -------------------- Helper to convert OpenIE5 kết quả ------------------

def _convert_openie5(raw_results: List[Dict]) -> List[Dict[str, str]]:
    """Chuyển đổi output gốc của pyopenie thành list các triple subject/relation/object."""
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


def terminate_corenlp_processes(port: int = 9000, silent: bool = True):
    """
    Find and terminate any running Stanford CoreNLP server processes.
    This is to ensure a fresh server starts with the correct properties.
    """
    if not silent:
        print(f"Checking for existing CoreNLP server processes on port {port}...")
    
    found_processes = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check for Java processes
            if 'java' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('stanford-corenlp' in arg for arg in cmdline):
                    # Check if it's running on the target port
                    port_arg = f'-port {port}'
                    if any(port_arg in arg for arg in cmdline) or (port_arg not in str(cmdline) and port == 9000):
                        if not silent:
                            print(f"  -> Found stray CoreNLP process {proc.pid}. Terminating...")
                        proc.terminate()  # Send SIGTERM
                        proc.wait(timeout=3) # Wait for graceful shutdown
                        found_processes = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue # Process may have died, or we don't have permissions
        except psutil.TimeoutExpired:
            if not silent:
                print(f"  -> Process {proc.pid} did not terminate gracefully. Forcing kill...")
            proc.kill() # Force kill
            found_processes = True

    if found_processes and not silent:
        print("Cleanup complete.")

# -------------------- Dừng OpenIE server ----------------------------------


def terminate_openie_processes(port: int = 9000, silent: bool = True):
    """Tìm và dừng các tiến trình OpenIE5 server đang chạy."""
    if not silent:
        print(f"Checking for existing OpenIE server processes on port {port}...")

    found_processes = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'java' in (proc.info['name'] or '').lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('openie-assembly' in arg for arg in cmdline):
                    # Kiểm tra port
                    port_arg = f'--httpPort {port}'
                    if any(port_arg in arg for arg in cmdline) or (port_arg not in str(cmdline) and port == 9000):
                        if not silent:
                            print(f"  -> Found OpenIE process {proc.pid}. Terminating...")
                        proc.terminate()
                        proc.wait(timeout=3)
                        found_processes = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except psutil.TimeoutExpired:
            if not silent:
                print(f"  -> Process {proc.pid} did not terminate gracefully. Forcing kill...")
            proc.kill()
            found_processes = True

    if found_processes and not silent:
        print("OpenIE cleanup complete.")

# Cấu hình mặc định cho CoreNLP - IMPROVED VERSION (removed harmful anti-cache settings)
DEFAULT_PROPERTIES = {
    "annotators": "tokenize,ssplit,pos,lemma,depparse,openie",
    "outputFormat": "json",
    "timeout": "600000",
    # Relax strict mode để CoreNLP sinh nhiều triple hơn
    "openie.triple.strict": "false",
    "openie.max_entailments_per_clause": "1000",
    # Giữ affinity cap thấp để không bỏ quan hệ tiềm năng
    "openie.affinity_probability_cap": "0.1",
}

# Hàm chính: tách câu và trích xuất quan hệ

def extract_relations_from_paragraph(
    paragraph_text: str,
    port: int = 9000,
    silent: bool = True
) -> List[Dict[str, str]]:
    if not paragraph_text.strip():
        return []

    results: List[Dict[str, str]] = []
    
    # Debug: In ra properties đang sử dụng (commented out for cleaner output)
    # print(f"Using properties: strict={properties.get('openie.triple.strict')}, "
    #       f"max_entailments={properties.get('openie.max_entailments_per_clause')}, "
    #       f"affinity_cap={properties.get('openie.affinity_probability_cap')}")
    
    # Import suppression if needed
    original_levels = {}
    if silent:
        import logging
        loggers_to_suppress = ['stanford', 'corenlp', 'stanza', 'java']
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.CRITICAL)
    
    try:
        # Nếu server chưa sẵn sàng, trả rỗng để tránh treo
        if not _is_port_open(port):
            return []

        client = _get_global_client(port=port)

        # Thêm timeout tạm cho socket để tránh treo lâu
        import socket as _socket
        prev_timeout = _socket.getdefaulttimeout()
        try:
            _socket.setdefaulttimeout(5.0)
            raw_results = client.extract(paragraph_text)
        finally:
            _socket.setdefaulttimeout(prev_timeout)
        # Chuyển đổi sang định dạng subject / relation / object
        results.extend(_convert_openie5(raw_results))
    except Exception as e:
        if not silent:
            print(f"Error in OIE extraction: {e}")
    finally:
        # Restore original logging levels if they were changed
        if silent and original_levels:
            for logger_name, original_level in original_levels.items():
                logger = logging.getLogger(logger_name)
                logger.setLevel(original_level)
    # Loại bỏ trùng lặp
    unique = []
    seen = set()
    # Giữ lại tất cả triple (chỉ khử trùng lặp tuyệt đối), tránh loại bỏ các quan hệ ngắn
    for r in results:
        key = f"{r['subject'].strip().lower()}|{r['relation'].strip().lower()}|{r['object'].strip().lower()}"
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# -------------------- TSV Processing Functions ---------------------------

def format_oie_triples_as_text(triples: List[Dict[str, str]]) -> str:
    """
    Format OIE triples as readable text for the raw_oie_data column.
    Each triple becomes: "(subject; relation; object)"
    """
    if not triples:
        return ""
    
    formatted_triples = []
    for triple in triples:
        subject = triple.get("subject", "").strip()
        relation = triple.get("relation", "").strip() 
        obj = triple.get("object", "").strip()
        
        if subject and relation and obj:
            formatted_triples.append(f"({subject}; {relation}; {obj})")
    
    return " ".join(formatted_triples)


def process_chunk_tsv_with_oie(
    input_tsv_path: str,
    output_tsv_path: str,
    chunk_text_column: str = "chunk_text",
    port: int = 9000,
    silent: bool = True,
    batch_size: int = 100
) -> None:
    """
    Process a TSV file with chunked data and add OIE extraction results.
    
    Args:
        input_tsv_path: Path to input TSV file with headers: query_id, document_id, chunk_text, label
        output_tsv_path: Path to output TSV file (will add raw_oie_data and raw_oie_data_plus_chunk_text columns)
        chunk_text_column: Name of column containing text to process with OIE
        port: Port for OpenIE5 server
        silent: Whether to suppress logging output
        batch_size: Number of rows to process at once
    """
    import pandas as pd
    import csv
    from tqdm import tqdm
    
    if not silent:
        print(f"Processing TSV file: {input_tsv_path}")
        print(f"Output will be saved to: {output_tsv_path}")
    
    # Read input TSV
    try:
        df = pd.read_csv(input_tsv_path, sep='\t', dtype=str)
    except Exception as e:
        raise ValueError(f"Error reading input TSV file: {e}")
    
    # Validate required columns
    required_columns = ["query_id", "document_id", chunk_text_column, "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not silent:
        print(f"Input file contains {len(df)} rows")
        print(f"Processing column: {chunk_text_column}")
    
    # Initialize new columns
    df["raw_oie_data"] = ""
    df["raw_oie_data_plus_chunk_text"] = ""
    
    # Process in batches to avoid memory issues and show progress
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    if not silent:
        print(f"Processing in {total_batches} batches of {batch_size} rows each...")
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches", disable=silent):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        
        for idx in range(start_idx, end_idx):
            chunk_text = df.loc[idx, chunk_text_column]
            
            # Skip empty or null text
            if pd.isna(chunk_text) or not str(chunk_text).strip():
                df.loc[idx, "raw_oie_data"] = ""
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = str(chunk_text) if not pd.isna(chunk_text) else ""
                continue
            
            try:
                # Extract OIE triples
                triples = extract_relations_from_paragraph(
                    str(chunk_text), 
                    port=port, 
                    silent=True
                )
                
                # Format OIE data as text
                oie_text = format_oie_triples_as_text(triples)
                
                # Set raw_oie_data column
                df.loc[idx, "raw_oie_data"] = oie_text
                
                # Set raw_oie_data_plus_chunk_text column (chunk + OIE)
                if oie_text.strip():
                    combined_text = f"{chunk_text} {oie_text}"
                else:
                    combined_text = str(chunk_text)
                
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = combined_text
                
            except Exception as e:
                if not silent:
                    print(f"Error processing row {idx}: {e}")
                # Keep original text if OIE fails
                df.loc[idx, "raw_oie_data"] = ""
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = str(chunk_text)
    
    # Save output TSV
    try:
        df.to_csv(output_tsv_path, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
        if not silent:
            print(f"Successfully saved output to: {output_tsv_path}")
            print(f"Output contains {len(df)} rows with {len(df.columns)} columns")
            print(f"New columns added: raw_oie_data, raw_oie_data_plus_chunk_text")
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
    """
    Alternative version that stores OIE data as JSON format for structured access.
    
    Args:
        input_tsv_path: Path to input TSV file
        output_tsv_path: Path to output TSV file  
        chunk_text_column: Name of column containing text to process
        port: Port for OpenIE5 server
        silent: Whether to suppress logging
        batch_size: Batch size for processing
    """
    import pandas as pd
    import csv
    import json
    from tqdm import tqdm
    
    if not silent:
        print(f"Processing TSV file with JSON OIE format: {input_tsv_path}")
    
    # Read input TSV
    try:
        df = pd.read_csv(input_tsv_path, sep='\t', dtype=str)
    except Exception as e:
        raise ValueError(f"Error reading input TSV file: {e}")
    
    # Validate required columns
    required_columns = ["query_id", "document_id", chunk_text_column, "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize new columns
    df["raw_oie_data_json"] = ""
    df["raw_oie_data_plus_chunk_text"] = ""
    
    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches", disable=silent):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        
        for idx in range(start_idx, end_idx):
            chunk_text = df.loc[idx, chunk_text_column]
            
            # Skip empty or null text
            if pd.isna(chunk_text) or not str(chunk_text).strip():
                df.loc[idx, "raw_oie_data_json"] = "[]"
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = str(chunk_text) if not pd.isna(chunk_text) else ""
                continue
            
            try:
                # Extract OIE triples
                triples = extract_relations_from_paragraph(
                    str(chunk_text), 
                    port=port, 
                    silent=True
                )
                
                # Store as JSON
                oie_json = json.dumps(triples, ensure_ascii=False)
                df.loc[idx, "raw_oie_data_json"] = oie_json
                
                # Create combined text with formatted triples
                oie_text = format_oie_triples_as_text(triples)
                if oie_text.strip():
                    combined_text = f"{chunk_text} {oie_text}"
                else:
                    combined_text = str(chunk_text)
                
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = combined_text
                
            except Exception as e:
                if not silent:
                    print(f"Error processing row {idx}: {e}")
                df.loc[idx, "raw_oie_data_json"] = "[]"
                df.loc[idx, "raw_oie_data_plus_chunk_text"] = str(chunk_text)
    
    # Save output
    try:
        df.to_csv(output_tsv_path, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
        if not silent:
            print(f"Successfully saved JSON format output to: {output_tsv_path}")
    except Exception as e:
        raise ValueError(f"Error saving output TSV file: {e}")


# -------------------- CLI Interface ---------------------------

def main():
    """Command line interface for OIE TSV processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process chunked TSV files with OpenIE extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python OIE.py input_chunks.tsv output_with_oie.tsv
  python OIE.py input_chunks.tsv output_with_oie.tsv --chunk-column chunk_text --batch-size 50
  python OIE.py input_chunks.tsv output_with_oie.tsv --json-format --port 9001
        """
    )
    
    parser.add_argument("input_tsv", help="Input TSV file path")
    parser.add_argument("output_tsv", help="Output TSV file path")
    parser.add_argument("--chunk-column", default="chunk_text", 
                       help="Name of column containing text to process (default: chunk_text)")
    parser.add_argument("--port", type=int, default=9000,
                       help="OpenIE5 server port (default: 9000)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing (default: 100)")
    parser.add_argument("--json-format", action="store_true",
                       help="Store OIE data as JSON instead of formatted text")
    parser.add_argument("--silent", action="store_true",
                       help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_tsv)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_tsv}' does not exist.")
        return 1
    
    # Create output directory if needed
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
                batch_size=args.batch_size
            )
        else:
            process_chunk_tsv_with_oie(
                input_tsv_path=args.input_tsv,
                output_tsv_path=args.output_tsv,
                chunk_text_column=args.chunk_column,
                port=args.port,
                silent=args.silent,
                batch_size=args.batch_size
            )
        
        if not args.silent:
            print("Processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
