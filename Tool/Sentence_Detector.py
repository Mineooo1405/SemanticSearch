"""
Sentence Detector and Simplifier
Splits text into paragraphs and sentences. Optionally simplifies complex sentences
for downstream tasks like Open Information Extraction (OIE) and Semantic Grouping.
"""

import re
from typing import List, Dict, Tuple, Optional, cast
import datetime
import os
import time # Thêm time để tạo timestamp nếu cần
import pickle
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

# --- NLTK Setup ---
_NLTK_DATA_DOWNLOADED = False

def _ensure_nltk_punkt():
    """
    Ensures that the NLTK 'punkt' tokenizer data is available,
    downloading it if necessary.
    """
    global _NLTK_DATA_DOWNLOADED
    if _NLTK_DATA_DOWNLOADED:
        return
    try:
        # Check for the general 'punkt' tokenizer data
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("[INFO] NLTK 'punkt' model not found. Downloading...")
        try:
            nltk.download('punkt', quiet=True)
            print("[INFO] 'punkt' model downloaded successfully.")
            _NLTK_DATA_DOWNLOADED = True
        except Exception as e:
            print(f"[ERROR] Failed to download NLTK 'punkt' model: {e}")
            print("[ERROR] Sentence tokenization might fail.")
    _NLTK_DATA_DOWNLOADED = True

def split_text_into_sentences(text: str) -> List[str]:
    """
    Splits a given text into sentences using NLTK's sentence tokenizer.
    This is generally more robust than regex-based methods.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []
        
    _ensure_nltk_punkt()
    
    try:
        # Explicitly load the tokenizer from its pickle file to bypass environment issues
        punkt_pickle_path = nltk.data.find('tokenizers/punkt/english.pickle')
        with open(punkt_pickle_path, 'rb') as f:
            punkt_tokenizer = cast(PunktSentenceTokenizer, pickle.load(f))
        
        return punkt_tokenizer.tokenize(text)
    except Exception as e:
        print(f"[ERROR] NLTK sentence tokenization failed: {e}")
        # Fallback to a very basic split as a last resort
        return [s.strip() for s in text.split('.') if s.strip()]

# Common abbreviations to protect during sentence splitting
COMMON_ABBREVIATIONS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "Gen.", "Col.", "Maj.", "Capt.",
    "Lt.", "St.", "Sr.", "Jr.", "Ph.D.", "M.D.", "B.A.", "M.A.", "i.e.", "e.g.",
    "etc.", "vs.", "a.m.", "p.m.", "U.S.", "U.K.", "E.U.", "Fig.", "No.", "Vol."
}

def protect_abbreviations(text: str) -> Tuple[str, Dict[str, str]]:
    """Replaces common abbreviations with placeholders to prevent incorrect splitting."""
    replacements = {}
    protected_text = text
    # Use regex for safer replacement (whole word only) might be better, but simple replace for now
    for abbr in COMMON_ABBREVIATIONS:
        placeholder = f"__ABBR_{len(replacements)}__"
        # Basic replacement, might replace parts of words if not careful
        protected_text = protected_text.replace(abbr, placeholder)
        replacements[placeholder] = abbr
    return protected_text, replacements

def restore_abbreviations(text: str, replacements: Dict[str, str]) -> str:
    """Restores abbreviations from placeholders."""
    result = text
    for placeholder, original in replacements.items():
        result = result.replace(placeholder, original)
    return result

def clean_text(text: str) -> str:
    """Cleans and normalizes raw text input."""
    # Normalize line breaks
    text = re.sub(r'\r\n|\r', '\n', text)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Handle special unicode line breaks
    text = text.replace('\u2028', '\n').replace('\u2029', '\n')
    # Remove list numbering at the start of lines
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Handle potentially problematic period usage like "word.1. Word"
    text = re.sub(r'(\d+)\.(\d+)\.', r'\1. \2.', text)
    text = re.sub(r'([a-zA-Z])\.(\d+)\.', r'\1. \2.', text, flags=re.IGNORECASE)
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace("'", "'")
    text = text.replace("—", "-").replace("–", "-")
    return text

def clean_text_for_nlp(text: str) -> str:
    """
    Cleans text by removing common metadata patterns that can confuse NLP sentence splitters like CoreNLP.
    """
    # Remove patterns like [Text], [Graphic], etc.
    text = re.sub(r'\[\w+\]', ' ', text)
    # Remove patterns like "Language: Spanish", "Article Type: BFN"
    text = re.sub(r'\b(Language|Article Type):\s*[\w\s]+\b', ' ', text, flags=re.IGNORECASE)
    # Normalize multiple dots or other sentence-ending characters to a single period
    text = re.sub(r'[.?!]{2,}', '.', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def normalize_paragraph(paragraph: str) -> str:
    """Normalizes a single paragraph by removing internal line breaks and standardizing whitespace."""
    # Remove internal line breaks
    clean_para = re.sub(r'\s*\n\s*', ' ', paragraph)
    # Standardize whitespace
    clean_para = re.sub(r'\s+', ' ', clean_para).strip()
    # Fix cases like "end.1. Start"
    clean_para = re.sub(r'(\.)(\d+\.\s)', r'\1 \2', clean_para)
    return clean_para

def split_into_paragraphs(text: str) -> List[str]:
    """Splits cleaned text into paragraphs based on double line breaks."""
    text = clean_text(text)
    # Rejoin hyphenated words broken across lines
    text = re.sub(r'([a-zA-Z])- *\n *([a-zA-Z])', r'\1\2', text)
    # Remove single line breaks used for formatting within paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Split into paragraphs based on double line breaks
    raw_paragraphs = text.split('\n\n')
    # Normalize each paragraph
    paragraphs = [normalize_paragraph(p) for p in raw_paragraphs if normalize_paragraph(p)]
    return paragraphs

def split_paragraph_into_sentences(paragraph: str) -> List[str]:
    """Splits a normalized paragraph into sentences, protecting abbreviations."""
    protected_text, replacements = protect_abbreviations(paragraph)

    # Split primarily by periods followed by space or end-of-string.
    # This is a basic approach; more robust splitting might consider ?, ! etc.
    # Using lookahead `(?=...)` to avoid consuming the delimiter space.
    raw_sentences = re.split(r'\.(?=\s+|$)', protected_text) # Basic split on '.'

    # Consider splitting on other punctuation like '?' and '!' as well
    # This requires a more complex regex or iterative splitting.
    # Example: re.split(r'(?<=[.?!])\s+', protected_text)

    sentences = []
    for sentence in raw_sentences:
        sentence = restore_abbreviations(sentence, replacements)
        sentence = sentence.strip()
        # Basic filter for empty strings or strings containing only numbers
        if sentence and not re.match(r'^\d+$', sentence):
            # Re-add the period if it's missing (common after split)
            if not re.search(r'[.!?]$', sentence):
                 sentence += "."
            # Ensure no internal newlines remain
            sentence = re.sub(r'\s*\n\s*', ' ', sentence)
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            if sentence: # Final check for empty string after cleaning
                sentences.append(sentence)
    return sentences

def _ensure_punctuation_and_capitalization(text: str) -> str:
    """Ensures the text ends with basic punctuation and starts with a capital letter."""
    text = text.strip()
    if not text:
        return ""

    # Capitalize the first letter
    if text[0].islower():
        if text.startswith(('"', "'")) and len(text) > 1:
            # Capitalize the first actual letter inside quotes
            text = text[0] + text[1].upper() + text[2:]
        else:
            text = text[0].upper() + text[1:]

    # Add a period if no ending punctuation is found
    if not re.search(r'[.!?]$', text):
        text += '.'
    return text

# Renamed from simplify_sentence_for_oie for broader applicability
def simplify_complex_sentence(sentence: str) -> List[str]:
    """
    Simplifies a complex sentence into potentially multiple simpler clauses using rule-based methods.
    This version is conservative, aiming to avoid creating incorrect fragments.
    (Version 3 - Conservative)
    """
    sentence = sentence.strip()
    # Use the combined function for ensuring format
    original_sentence_with_punct = _ensure_punctuation_and_capitalization(sentence)

    # Remove ending punctuation for internal processing
    is_question = sentence.endswith('?')
    if sentence.endswith(('.', '?')):
        sentence = sentence[:-1].strip()

    MIN_WORDS_FOR_SIMPLIFICATION = 6
    if not sentence or len(sentence.split()) < MIN_WORDS_FOR_SIMPLIFICATION:
        return [original_sentence_with_punct]

    simplified_clauses = []

    # Rule 0: Do not split questions
    wh_words = ["who", "what", "where", "when", "why", "how", "which"]
    if is_question or any(sentence.lower().startswith(wh + " ") for wh in wh_words):
        return [original_sentence_with_punct]

    # Rule 1: Split by semicolon if both parts seem clausal (>= 3 words)
    clauses_sc = [c.strip() for c in sentence.split(';') if c.strip()]
    if len(clauses_sc) > 1 and all(len(c.split()) >= 3 for c in clauses_sc):
        # Avoid splitting if the second part is just a connector adverb
        if len(clauses_sc[1].split()) == 1 and clauses_sc[1].lower() in ['alternatively', 'however', 'therefore', 'moreover']:
             pass # Do not split
        else:
            # No deep recursion here to keep it simple and avoid errors
            simplified_clauses.extend(clauses_sc)
            if len(simplified_clauses) > 1:
                 # Apply final formatting to each part
                 return [_ensure_punctuation_and_capitalization(s) for s in simplified_clauses if s]

    # Rule 2: Non-restrictive relative clauses (after comma + which/who)
    match_rel = re.search(r'(?P<main>.*?),\s+(?P<rel_pron>which|who)\s+(?P<rel_clause>[^"]*)', sentence, re.IGNORECASE)
    if match_rel:
        main_part = match_rel.group('main').strip()
        rel_clause_content = match_rel.group('rel_clause').strip()
        rel_pron = match_rel.group('rel_pron')
        main_words = main_part.split()
        antecedent_phrase = ""
        # Try to get a noun phrase as antecedent
        if len(main_words) >= 2 and main_words[-2].lower() in ["a", "an", "the", "his", "her", "its", "their", "some", "many", "this", "that", "these", "those"]:
             antecedent_phrase = " ".join(main_words[-2:])
        elif main_words:
             antecedent_phrase = main_words[-1]
        if antecedent_phrase and antecedent_phrase.endswith((',', ';', ':')):
            antecedent_phrase = antecedent_phrase[:-1].strip()

        if antecedent_phrase:
            reconstructed_rel = ""
            rel_clause_words = rel_clause_content.split()
            verb_starts = ['is', 'was', 'are', 'were', 'has', 'had', 'can', 'could', 'may', 'might', 'will', 'would', 'should']
            # Reconstruct only if it seems grammatically plausible
            if rel_clause_words and rel_clause_words[0].lower() in verb_starts:
                 reconstructed_rel = f"{antecedent_phrase} {rel_clause_content}"
            elif rel_clause_words and rel_pron.lower() == 'who' and len(rel_clause_words) > 1:
                 reconstructed_rel = f"{antecedent_phrase} {rel_clause_content}"

            # Split only if reconstructed part is valid and main part is long enough
            if reconstructed_rel and len(reconstructed_rel.split()) >= 3 and len(main_part.split()) >= 3:
                 simplified_clauses.append(main_part)
                 simplified_clauses.append(reconstructed_rel)
                 return [_ensure_punctuation_and_capitalization(s) for s in simplified_clauses if s]
            else:
                 # Fallback: return only the main part if reconstruction is uncertain
                 simplified_clauses.append(main_part)
                 return [_ensure_punctuation_and_capitalization(s) for s in simplified_clauses if s]

    # Rule 3: Coordinating conjunctions after a comma (and, but, or)
    parts_coord = re.split(r'(,\s*(?:and|but|or)\s+)', sentence, maxsplit=1)
    if len(parts_coord) == 3:
        clause1 = parts_coord[0].strip()
        clause2 = parts_coord[2].strip()
        # Split only if both parts seem like valid clauses
        if len(clause1.split()) >= 3 and len(clause2.split()) >= 3:
            simplified_clauses.append(clause1)
            simplified_clauses.append(clause2)
            return [_ensure_punctuation_and_capitalization(s) for s in simplified_clauses if s]

    # Rule 4: Nested 'that' clauses with 'or that'
    match_nested_that = re.search(r'^(?P<prefix>.*\s(?:maintain|believe|suggest|think|argue|state|say|show)\s+that\s+)(?P<clauseA>.*?)\s+or\s+that\s+(?P<clauseB>.*)$', sentence, re.IGNORECASE)
    if match_nested_that:
        prefix = match_nested_that.group('prefix').strip()
        clauseA = match_nested_that.group('clauseA').strip()
        clauseB = match_nested_that.group('clauseB').strip()
        if clauseA and clauseB:
             # Create two separate sentences repeating the prefix
             simplified_clauses.append(f"{prefix} {clauseA}")
             simplified_clauses.append(f"{prefix} {clauseB}")
             return [_ensure_punctuation_and_capitalization(s) for s in simplified_clauses if s]

    # If no rules applied, return the original sentence, properly formatted
    return [original_sentence_with_punct]

# Renamed from apply_oie_simplification
def apply_sentence_simplification(sentences: List[str]) -> List[str]:
    """Applies simplification rules to a list of sentences and filters the results."""
    simplified_list = []
    for sentence in sentences:
        # Apply simplification rules once per sentence
        simplified_list.extend(simplify_complex_sentence(sentence))

    # Post-processing: remove duplicates, very short sentences, ensure formatting
    final_list = []
    seen = set()
    MIN_WORDS_FINAL = 4 # Minimum words for a sentence to be kept

    for s in simplified_list:
        # Ensure punctuation and capitalization
        s_formatted = _ensure_punctuation_and_capitalization(s)
        words = s_formatted.split()

        if len(words) >= MIN_WORDS_FINAL and s_formatted not in seen:
             # Filter out sentences that are just standalone conjunctions or markers
             if not re.match(r'^(And|But|Or|However|Therefore|Moreover)\.$', s_formatted, re.IGNORECASE):
                final_list.append(s_formatted)
                seen.add(s_formatted)
    return final_list

# Renamed from detect_sentences
def extract_and_simplify_sentences(text: str, simplify: bool = False) -> List[str]:
    """
    Top-level function to extract and optionally simplify sentences from a raw text string.
    """
    if not text or not isinstance(text, str):
        return []

    # --- ADDED: Clean text at the very beginning ---
    cleaned_text = clean_text_for_nlp(text)
    
    paragraphs = split_into_paragraphs(cleaned_text)
    all_sentences = []
    
    for para in paragraphs:
        all_sentences.extend(split_paragraph_into_sentences(para))

    # Basic cleaning
    clean_initial_sentences = []
    for sentence in all_sentences:
        clean_sentence = re.sub(r'\s*\n\s*', ' ', sentence).strip()
        if clean_sentence:
            # Ensure basic format even if not simplifying
            clean_initial_sentences.append(_ensure_punctuation_and_capitalization(clean_sentence))

    if simplify:
        # Apply simplification rules and filtering
        final_sentences = apply_sentence_simplification(clean_initial_sentences)
    else:
        final_sentences = clean_initial_sentences # Already formatted

    # Final check for empty strings
    return [s for s in final_sentences if s]

# --- UI Functions ---

def get_text_from_user() -> str:
    """Gets text input directly from the user."""
    print("\nNhập nội dung văn bản (kết thúc bằng một dòng chỉ có '###'):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == '###':
                break
            lines.append(line)
        except EOFError: # Handle Ctrl+D/Ctrl+Z
            break
    return "\n".join(lines)

def get_text_from_file() -> Tuple[Optional[str], Optional[str]]:
    """Gets text input from a file specified by the user."""
    file_path = input("\nNhập đường dẫn đến file văn bản: ")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"Đã tải thành công văn bản từ {file_path}")
        # Lấy tên file làm document_id
        document_id = os.path.splitext(os.path.basename(file_path))[0]
        return text, document_id
    except FileNotFoundError:
        print(f"\nError: File not found at '{file_path}'. Please check the path.")
        return None, None
    except Exception as e:
        print(f"\nAn unexpected error occurred while reading the file: {e}")
        return None, None

def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    """Asks a yes/no question and returns a boolean."""
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        response = input(f"{prompt} {suffix}: ").strip().lower()
        if not response:
            return default_yes
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Vui lòng trả lời 'y' hoặc 'n'.")

def process_and_output(text: str, document_id: str, simplify: bool):
    """Processes the text and handles output based on user choice."""
    if not text or not text.strip():
        print("Không có văn bản nào để xử lý.")
        return

    print("\nĐang xử lý văn bản...")
    start_time = datetime.datetime.now()

    # --- Sentence Extraction and Simplification ---
    all_sentences = extract_and_simplify_sentences(text, simplify=simplify)

    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"Xử lý hoàn tất trong {processing_time:.4f} giây.")
    print(f"Tìm thấy {len(all_sentences)} câu{' (đã đơn giản hóa)' if simplify else ''}.")

    # --- Output Handling ---
    save_output = ask_yes_no("\nBạn có muốn lưu kết quả ra file không?", default_yes=True)

    if save_output:
        default_filename = f"sentences_{document_id}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        output_filename = input(f"Nhập tên file đầu ra (mặc định: {default_filename}): ").strip()
        if not output_filename:
            output_filename = default_filename

        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Đã tạo thư mục: {output_dir}")
            except OSError as e:
                 print(f"Lỗi khi tạo thư mục đầu ra '{output_dir}': {e}")
                 # Continue, try writing in current directory

        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                # Ghi thông tin cơ bản
                f.write(f"===== KẾT QUẢ TÁCH CÂU =====\n\n")
                f.write(f"Tài liệu: {document_id}\n")
                f.write(f"Thời gian xử lý: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Thời gian thực hiện: {processing_time:.4f} giây\n")
                f.write(f"Chế độ đơn giản hóa: {'Bật' if simplify else 'Tắt'}\n\n")
                f.write(f"Tổng số câu: {len(all_sentences)}\n")
                f.write(f"Độ dài văn bản gốc: {len(text)} ký tự\n\n")
                f.write(f"===== DANH SÁCH CÂU =====\n\n")

                for i, sentence in enumerate(all_sentences):
                    f.write(f"{i+1}. {sentence}\n")

            print(f"\nĐã lưu kết quả vào file: {output_filename}")

        except IOError as e:
            print(f"\nLỗi khi ghi vào tệp đầu ra '{output_filename}': {e}")
            print("\n--- Kết quả (in ra console do lỗi ghi file) ---")
            for i, sentence in enumerate(all_sentences):
                print(f"{i+1}. {sentence}")
    else:
        # Hiển thị trên console
        print(f"\n--- Kết quả ---")
        for i, sentence in enumerate(all_sentences):
            print(f"{i+1}. {sentence}")

# --- Main Menu ---
def main_menu():
    """Displays the main menu and handles user interaction."""
    while True:
        print("\n=== SENTENCE DETECTOR & SIMPLIFIER ===")
        print("1. Nhập văn bản trực tiếp")
        print("2. Tải văn bản từ file")
        print("3. Thoát")

        choice = input("\nLựa chọn của bạn: ").strip()

        text = None
        document_id = "manual_input" # ID mặc định cho nhập trực tiếp

        if choice == '1':
            text = get_text_from_user()
        elif choice == '2':
            text, doc_id_from_file = get_text_from_file()
            if doc_id_from_file: # Nếu tải file thành công, dùng tên file làm ID
                document_id = doc_id_from_file
        elif choice == '3':
            print("Đang thoát...")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng chọn lại.")
            continue

        # Nếu có text để xử lý
        if text is not None:
            simplify_choice = ask_yes_no("\nBật chế độ đơn giản hóa câu?", default_yes=False)
            process_and_output(text, document_id, simplify=simplify_choice)
        else:
            # Nếu get_text_from_file trả về None (lỗi đọc file)
            print("Không thể xử lý do lỗi nhập liệu.")


# Main execution block
if __name__ == "__main__":
    main_menu()
