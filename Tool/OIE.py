import os
import sys
import logging
import psutil
from typing import List, Dict, Optional
from stanfordnlp.server import CoreNLPClient

# Suppress CoreNLP and Java startup logging
logging.getLogger("stanford").setLevel(logging.WARNING)
logging.getLogger("corenlp").setLevel(logging.WARNING)

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

# Cấu hình mặc định cho CoreNLP - IMPROVED VERSION (removed harmful anti-cache settings)
DEFAULT_PROPERTIES = {
    "annotators": "tokenize,ssplit,pos,lemma,depparse,openie",
    "outputFormat": "json",
    "timeout": "600000",
    "openie.triple.strict": "false",
    "openie.max_entailments_per_clause": "800",  
    "openie.affinity_probability_cap": "0.3"
}

# Hàm chính: tách câu và trích xuất quan hệ

def extract_relations_from_paragraph(
    paragraph_text: str,
    port: int = 9000,
    silent: bool = True
) -> List[Dict[str, str]]:
    # --- NEW: Terminate stray processes before starting --- #
    terminate_corenlp_processes(port=port, silent=silent)

    if not paragraph_text.strip():
        return []
    
    # Luôn sử dụng ENHANCED_PROPERTIES
    properties = DEFAULT_PROPERTIES
    
    results = []
    
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
        # Quay lại cách khởi tạo ổn định bằng properties
        with CoreNLPClient(
            properties=properties,
            timeout=int(properties.get("timeout", "600000")),
            memory='8G',
            be_quiet=True,
            start_server=True,
            endpoint=f"http://localhost:{port}"
        ) as client:
            # Annotate toàn bộ đoạn văn bản một lần, CoreNLP sẽ tự tách câu
            ann = client.annotate(paragraph_text)
            # Lặp qua các câu đã annotate (ann là dictionary)
            if 'sentences' in ann:
                for sentence_ann in ann['sentences']:
                    # Lặp qua các OpenIE triples
                    if 'openie' in sentence_ann:
                        for triple in sentence_ann['openie']:
                            results.append({
                                "subject": triple.get("subject", ""),
                                "relation": triple.get("relation", ""),
                                "object": triple.get("object", "")
                            })
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
    for r in results:
        key = f"{r['subject'].strip().lower()}|{r['relation'].strip().lower()}|{r['object'].strip().lower()}"
        if key not in seen and all(len(r[k].strip()) > 1 for k in r):
            seen.add(key)
            unique.append(r)
    return unique
