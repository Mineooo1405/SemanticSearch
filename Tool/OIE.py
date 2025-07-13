import logging
import psutil
import subprocess
import socket
import time
from pathlib import Path
from typing import List, Dict, Optional
from pyopenie import OpenIE5  # type: ignore

# Suppress CoreNLP and Java startup logging
logging.getLogger("stanford").setLevel(logging.WARNING)
logging.getLogger("corenlp").setLevel(logging.WARNING)

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
        "-Xms8g",          # heap khởi đầu 8 GB
        "-Xmx12g",         # heap tối đa 12 GB  (đủ mà dư 8 GB cho Python)
        "-XX:+UseG1GC",    # GC G1 – ổn định trên heap lớn
        "-XX:MaxGCPauseMillis=200",
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
        client = _get_global_client(port=port)

        raw_results = client.extract(paragraph_text)
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
