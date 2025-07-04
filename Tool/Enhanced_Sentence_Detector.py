"""
Enhanced Sentence Detector for Semantic Analysis

Focuses on breaking text into atomic semantic units for better analysis.
- Handles contractions correctly 
- Preserves quotations
- Prevents fragment splitting
- Avoids duplicate content
- Maintains parenthetical clauses
"""

import re
import spacy
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_sentence_detector")

# Global model cache
_MODEL_CACHE = {}

# Common contractions to protect
CONTRACTIONS = {
    "'s", "'re", "'ve", "'ll", "'d", "'m", "n't", "'t"
}

# Words that often signal a fragment when at the start of a sentence
FRAGMENT_STARTERS = {
    "and", "or", "but", "yet", "so", "for", "nor", "because", "although", 
    "though", "since", "unless", "while", "whereas", "if", "until", "as",
    "however", "therefore", "thus", "moreover", "furthermore", "nevertheless",
    "which", "who", "when", "where", "why", "how", "that"
}

# Common speaking verbs used in attributions
ATTRIBUTION_VERBS = {
    "says", "said", "asks", "asked", "explains", "explained", "notes", "noted",
    "cautions", "cautioned", "continues", "continued", "adds", "added"
}

def load_model(model_name="en_core_web_trf"):
    """Load spaCy model with caching"""
    if (model_name in _MODEL_CACHE):
        return _MODEL_CACHE[model_name]
    
    try:
        nlp = spacy.load(model_name)
        # Optimize pipeline for sentence segmentation
        if "ner" in nlp.pipe_names:
            nlp.disable_pipe("ner")
        _MODEL_CACHE[model_name] = nlp
        return nlp
    except OSError:
        logger.info(f"Downloading {model_name}...")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
            _MODEL_CACHE[model_name] = nlp
            return nlp
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            logger.info("Falling back to en_core_web_sm")
            try:
                if "en_core_web_sm" not in _MODEL_CACHE:
                    spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                _MODEL_CACHE["en_core_web_sm"] = nlp
                return nlp
            except Exception as e:
                logger.error(f"Failed to load fallback model: {e}")
                raise ValueError("No spaCy model available")

def protect_contractions_and_quotes(text: str) -> Tuple[str, Dict[str, str]]:
    """Protect contractions and quotes from being split incorrectly"""
    replacements = {}
    protected_text = text
    
    # First, preserve quoted text to avoid splitting it
    # Find all quote blocks (both single and double quotes)
    quote_patterns = [
        r'(["\']).+?\1',  # Basic quote pattern
        r'["\'"].*?["\'"]',  # Smart quotes
    ]
    
    for pattern in quote_patterns:
        for match_idx, match in enumerate(re.finditer(pattern, protected_text)):
            quote_text = match.group(0)
            placeholder = f"__QUOTE_{len(replacements)}__"
            replacements[placeholder] = quote_text
            protected_text = protected_text.replace(quote_text, placeholder, 1)
    
    # Now replace contractions with placeholders
    for contraction in CONTRACTIONS:
        pattern = r'(\w+)(' + re.escape(contraction) + r')\b'
        
        def replace_contraction(match):
            word = match.group(1) + match.group(2)
            placeholder = f"__CONTR_{len(replacements)}__"
            replacements[placeholder] = word
            return placeholder
        
        protected_text = re.sub(pattern, replace_contraction, protected_text)
    
    return protected_text, replacements

def restore_protected_text(text: str, replacements: Dict[str, str]) -> str:
    """Restore protected contractions and quotes"""
    result = text
    
    # Sort keys by length in descending order to avoid partial replacements
    keys = sorted(replacements.keys(), key=len, reverse=True)
    
    for placeholder in keys:
        result = result.replace(placeholder, replacements[placeholder])
    
    return result

def clean_text(text: str) -> Tuple[str, Dict[str, str]]:
    """Clean and normalize text for better processing"""
    # Standardize line breaks
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Handle the case with numbered lines (common in your input)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('’', "'").replace('’', "'")

    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    
    # Fix spacing after punctuation
    text = re.sub(r'([,.;:!?])(?=[^\s])', r'\1 ', text)
    
    # Remove duplicate periods
    text = re.sub(r'\.\.+', '.', text)
    
    # Protect contractions and quotes
    text, replacements = protect_contractions_and_quotes(text)
    
    return text, replacements

def preprocess_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs and clean each paragraph"""
    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Process each paragraph
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Remove duplicate sentences within paragraphs
        sentences = []
        seen = set()
        
        # Simple sentence splitting for deduplication
        for sentence in re.split(r'(?<=[.!?])\s+', paragraph):
            normalized = re.sub(r'\s+', ' ', sentence.lower())
            if normalized and normalized not in seen:
                seen.add(normalized)
                sentences.append(sentence)
        
        cleaned_para = ' '.join(sentences)
        cleaned_paragraphs.append(cleaned_para)
    
    return cleaned_paragraphs

def is_complete_sentence(sentence: str, nlp) -> bool:
    """Check if a string forms a complete sentence grammatically"""
    # Clean up the sentence for analysis
    cleaned = sentence.strip()
    if not cleaned or len(cleaned.split()) < 2:
        return False
    
    # Process with spaCy
    doc = nlp(cleaned)
    
    # A complete sentence needs a subject and a root verb (predicate)
    has_subject = any(token.dep_.startswith("nsubj") for token in doc)
    has_predicate = any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in doc)
    
    # Check for imperative sentences (commands) which may not have explicit subjects
    is_imperative = len(doc) > 0 and doc[0].pos_ == "VERB"
    
    # A sentence is complete if it has both subject and predicate, or if it's an imperative
    return (has_subject and has_predicate) or is_imperative

def is_likely_attribution(sentence: str) -> bool:
    """Check if a string appears to be a quote attribution"""
    # Common patterns like "he said", "said John", etc.
    sentence = sentence.strip().lower()
    
    # Check for attribution verbs
    for verb in ATTRIBUTION_VERBS:
        if verb in sentence.split():
            return True
    
    # Common attribution patterns
    attribution_patterns = [
        r'\w+\s+(said|says|asked|noted|explained|continued|added)',
        r'(said|says|asked|noted|explained|continued|added)\s+\w+',
        r'according\s+to\s+\w+',
    ]
    
    for pattern in attribution_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    
    return False

def needs_joining_with_next(sentence: str, next_sentence: str = None) -> bool:
    """Determine if a sentence should be joined with the next one"""
    if not next_sentence:
        return False
    
    # If current sentence ends with a quote and next is an attribution
    if sentence.endswith('"') and is_likely_attribution(next_sentence):
        return True
    
    # If current sentence ends with an open quote
    if sentence.count('"') % 2 == 1 or sentence.count("'") % 2 == 1:
        return True
    
    # If next sentence starts with a lowercase letter (unusual for sentence start)
    if next_sentence and next_sentence[0].islower() and not next_sentence.startswith(("'", '"')):
        return True
    
    # If next sentence starts with a fragment marker
    first_word = next_sentence.split()[0].lower() if next_sentence.split() else ""
    if first_word in FRAGMENT_STARTERS:
        return True
    
    return False

def join_sentences(sentences: List[str]) -> List[str]:
    """Join sentences that need to be combined"""
    if not sentences:
        return []
    
    result = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i]
        
        # Check if we need to join with the next sentence
        if i < len(sentences) - 1 and needs_joining_with_next(current, sentences[i + 1]):
            # Join current and next sentences
            combined = current
            
            # Determine how to join them
            next_sent = sentences[i + 1]
            
            if current.endswith('"') and is_likely_attribution(next_sent):
                # Quote followed by attribution - just add a space
                combined = f"{current} {next_sent}"
            elif current.endswith(('.', '!', '?', ':', ';', ',')) or next_sent.startswith(('.', '!', '?', ':', ';', ',')):
                # If there's already punctuation, just add a space
                combined = f"{current} {next_sent}"
            else:
                # Add comma between them
                combined = f"{current}, {next_sent}"
            
            result.append(combined)
            i += 2  # Skip the next sentence since we've used it
        else:
            # Add the current sentence as is
            result.append(current)
            i += 1
    
    return result

def handle_quotes(sentences: List[str]) -> List[str]:
    """Ensure quotes are properly handled"""
    result = []
    quote_buffer = []
    in_quote = False
    
    for sentence in sentences:
        # Check for an odd number of quote marks
        double_quotes = sentence.count('"')
        single_quotes = sentence.count("'")
        
        if in_quote:
            # Already collecting a multi-sentence quote
            quote_buffer.append(sentence)
            
            # Check if this sentence closes the quote
            if double_quotes % 2 == 1 or single_quotes % 2 == 1:
                in_quote = False
                # Join all buffered sentences
                result.append(" ".join(quote_buffer))
                quote_buffer = []
        else:
            # Check if this starts a new quote
            if (double_quotes % 2 == 1 or single_quotes % 2 == 1):
                # This could be a multi-sentence quote
                if (sentence.startswith('"') or sentence.startswith("'")):
                    in_quote = True
                    quote_buffer = [sentence]
                else:
                    # Handle quote that doesn't start at the beginning
                    # Find the beginning of the quote
                    quote_pos = min(
                        sentence.find('"') if sentence.find('"') != -1 else float('inf'),
                        sentence.find("'") if sentence.find("'") != -1 else float('inf')
                    )
                    
                    if quote_pos < len(sentence) - 1:
                        # Split at the quote
                        before_quote = sentence[:quote_pos].strip()
                        quote_part = sentence[quote_pos:].strip()
                        
                        if before_quote:
                            result.append(before_quote)
                        
                        in_quote = True
                        quote_buffer = [quote_part]
                    else:
                        # Quote is at the end, treat normally
                        result.append(sentence)
            else:
                # Normal sentence, no unclosed quotes
                result.append(sentence)
    
    # Add any remaining quote buffer
    if quote_buffer:
        result.append(" ".join(quote_buffer))
    
    return result

def filter_sentences(sentences: List[str], nlp) -> List[str]:
    """Filter and clean the final list of sentences"""
    if not sentences:
        return []
    
    # Filter out empty or very short sentences
    non_empty = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 2]
    
    # Remove duplicate or very similar sentences
    unique_sentences = []
    seen_content = set()
    
    for sentence in non_empty:
        # Create normalized version for comparison
        normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        
        # Skip if very similar to existing content
        if normalized in seen_content:
            continue
            
        # Check similarity with existing sentences
        is_similar = False
        for existing in unique_sentences:
            existing_norm = re.sub(r'\s+', ' ', existing.lower().strip())
            existing_norm = re.sub(r'[^\w\s]', '', existing_norm)
            
            # Check if one is subset of the other
            if normalized in existing_norm or existing_norm in normalized:
                is_similar = True
                break
                
            # Check similarity ratio
            if len(normalized) > 10 and len(existing_norm) > 10:
                similarity = similarity_ratio(normalized, existing_norm)
                if similarity > 0.8:  # 80% similar
                    is_similar = True
                    break
        
        if not is_similar:
            seen_content.add(normalized)
            unique_sentences.append(sentence)
    
    # Ensure sentences are grammatically complete
    complete_sentences = []
    for sentence in unique_sentences:
        # Ensure sentence ends with proper punctuation
        if not sentence.endswith(('.', '!', '?', '"', "'")):
            sentence = sentence + '.'
        
        # For quotes, don't strictly check completeness
        if sentence.startswith('"') or sentence.startswith("'"):
            complete_sentences.append(sentence)
        # Skip very short sentences that are likely fragments
        elif len(sentence.split()) < 3 and not is_complete_sentence(sentence, nlp):
            continue
        else:
            complete_sentences.append(sentence)
    
    return complete_sentences

def similarity_ratio(str1: str, str2: str) -> float:
    """Calculate similarity between two strings"""
    # Use SequenceMatcher for similarity
    return SequenceMatcher(None, str1, str2).ratio()

def extract_sentences(text: str, model_name: str = "en_core_web_trf") -> List[str]:
    """
    Extract well-formed sentences from text, optimized for semantic analysis.
    
    Args:
        text: Input text
        model_name: Name of spaCy model to use
        
    Returns:
        List of well-formed sentences
    """
    # Clean and preprocess text
    cleaned_text, replacements = clean_text(text)
    paragraphs = preprocess_paragraphs(cleaned_text)
    
    # Load NLP model
    nlp = load_model(model_name)
    
    # Process each paragraph to extract sentences
    all_sentences = []
    
    for paragraph in paragraphs:
        # Use spaCy's sentence detector as a starting point
        doc = nlp(paragraph)
        initial_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Restore protected text (quotes and contractions)
        restored_sentences = [restore_protected_text(s, replacements) for s in initial_sentences]
        
        # Join sentences that need to be combined
        joined_sentences = join_sentences(restored_sentences)
        
        # Handle multi-sentence quotes
        quoted_sentences = handle_quotes(joined_sentences)
        
        # Add to overall list
        all_sentences.extend(quoted_sentences)
    
    # Final filtering and cleaning
    final_sentences = filter_sentences(all_sentences, nlp)
    
    return final_sentences

def detect_sentences(text: str, model_name: str = "en_core_web_trf") -> List[str]:
    """
    Public API for extracting sentences optimized for semantic analysis.
    
    Args:
        text: Input text
        model_name: Name of spaCy model to use
        
    Returns:
        List of sentences for semantic analysis
    """
    return extract_sentences(text, model_name)

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Sentence Detector")
    parser.add_argument("--input", "-i", help="Input text file")
    parser.add_argument("--output", "-o", help="Output file for sentences")
    parser.add_argument("--model", "-m", default="en_core_web_trf", 
                      help="Model to use (default: en_core_web_trf)")
    
    args = parser.parse_args()
    
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Enter text (type 'exit' on a line by itself to quit):")
        lines = []
        while True:
            try:
                line = input()
                if line.lower() == 'exit':
                    break
                lines.append(line)
            except EOFError:
                break
        text = '\n'.join(lines)
    
    if not text.strip():
        print("No text provided.")
        exit(1)
    
    # Process the text
    sentences = detect_sentences(text, args.model)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences):
                f.write(f"{i+1}. {sentence}\n")
        print(f"Wrote {len(sentences)} sentences to {args.output}")
    else:
        print(f"\nFound {len(sentences)} sentences:")
        for i, sentence in enumerate(sentences):
            print(f"{i+1}. {sentence}")