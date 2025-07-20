"""
Enhanced sentence segmentation using spaCy pre-trained models
Replaces custom sentence detector with robust NLP library
"""

import spacy
from typing import List, Optional
import re
import os
import warnings

# Global variables for model management
_nlp_model = None
_model_name = "en_core_web_sm"  # Default English model

def _ensure_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """
    Ensure spaCy model is installed and loaded
    
    Args:
        model_name: spaCy model name (default: en_core_web_sm)
        
    Returns:
        bool: True if model is available, False otherwise
    """
    global _nlp_model, _model_name
    
    if _nlp_model is not None and _model_name == model_name:
        return True
    
    try:
        # Try to load the model
        _nlp_model = spacy.load(model_name)
        _model_name = model_name
        
        # Disable unnecessary components for performance
        # Only keep essential components for sentence segmentation
        disable_pipes = []
        available_pipes = _nlp_model.pipe_names
        
        # Disable heavy components we don't need for sentence segmentation
        for pipe in ['ner', 'lemmatizer', 'textcat', 'textcat_multilabel']:
            if pipe in available_pipes:
                disable_pipes.append(pipe)
        
        if disable_pipes:
            _nlp_model.disable_pipes(disable_pipes)
        
        print(f"[INFO] Loaded spaCy model: {model_name}")
        return True
        
    except OSError:
        print(f"[WARNING] spaCy model '{model_name}' not found")
        
        # Try to download the model automatically
        try:
            import subprocess
            import sys
            
            print(f"[INFO] Attempting to download {model_name}...")
            result = subprocess.run([
                sys.executable, '-m', 'spacy', 'download', model_name
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"[INFO] Successfully downloaded {model_name}")
                _nlp_model = spacy.load(model_name)
                _model_name = model_name
                
                # Disable heavy components we don't need for sentence segmentation
                disable_pipes = []
                available_pipes = _nlp_model.pipe_names
                
                for pipe in ['ner', 'lemmatizer', 'textcat', 'textcat_multilabel']:
                    if pipe in available_pipes:
                        disable_pipes.append(pipe)
                
                if disable_pipes:
                    _nlp_model.disable_pipes(disable_pipes)
                    
                return True
            else:
                print(f"[ERROR] Failed to download {model_name}: {result.stderr}")
                
        except Exception as e:
            print(f"[ERROR] Auto-download failed: {e}")
        
        # Fallback to smaller model
        if model_name != "en_core_web_sm":
            print("[INFO] Trying fallback model: en_core_web_sm")
            return _ensure_spacy_model("en_core_web_sm")
        
        return False
    
    except Exception as e:
        print(f"[ERROR] Failed to load spaCy model: {e}")
        return False

def _fallback_sentence_split(text: str) -> List[str]:
    """
    Fallback sentence splitting using regex patterns
    Used when spaCy is not available
    """
    if not text or not isinstance(text, str):
        return []
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Basic sentence splitting regex
    # Split on . ! ? followed by whitespace and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Filter and clean
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) >= 10:  # Minimum viable sentence
            # Ensure sentence ends with punctuation
            if not re.search(r'[.!?]$', sentence):
                sentence += '.'
            result.append(sentence)
    
    return result

def extract_sentences_spacy(text: str, max_sent_length: int = 1000) -> List[str]:
    """
    Extract sentences using spaCy's sentence segmentation
    
    Args:
        text: Input text to segment
        max_sent_length: Maximum length for individual sentences
        
    Returns:
        List of sentence strings
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []
    
    # Try to use spaCy model
    if not _ensure_spacy_model():
        print("[WARNING] spaCy not available, using fallback method")
        return _fallback_sentence_split(text)
    
    try:
        # Process text with spaCy
        doc = _nlp_model(text)
        
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Filter very short sentences
            if len(sent_text) < 10:
                continue
            
            # Handle very long sentences
            if len(sent_text) > max_sent_length:
                # Split long sentences at punctuation marks
                sub_sentences = re.split(r'(?<=[.!?;])\s+', sent_text)
                for sub_sent in sub_sentences:
                    sub_sent = sub_sent.strip()
                    if len(sub_sent) >= 10:
                        if not re.search(r'[.!?]$', sub_sent):
                            sub_sent += '.'
                        sentences.append(sub_sent)
            else:
                # Ensure proper punctuation
                if not re.search(r'[.!?]$', sent_text):
                    sent_text += '.'
                sentences.append(sent_text)
        
        return sentences
        
    except Exception as e:
        print(f"[ERROR] spaCy processing failed: {e}")
        return _fallback_sentence_split(text)

def count_tokens_spacy(text: str) -> int:
    """
    Count tokens using spaCy tokenizer
    More accurate than simple word splitting
    """
    if not text or not isinstance(text, str):
        return 0
    
    if not _ensure_spacy_model():
        # Fallback to regex-based counting
        import re
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.strip())
        return len(tokens)
    
    try:
        doc = _nlp_model(text)
        return len([token for token in doc if not token.is_space])
    except Exception:
        # Fallback
        import re
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.strip())
        return len(tokens)

def to_sentences(text: str) -> List[str]:
    """
    Main sentence extraction function - drop-in replacement
    Compatible with existing chunking methods
    """
    return extract_sentences_spacy(text, max_sent_length=500)

def extract_and_simplify_sentences(text: str, simplify: bool = False) -> List[str]:
    """
    Extract sentences with optional simplification
    Compatible with existing Sentence_Detector interface
    """
    sentences = extract_sentences_spacy(text)
    
    if not simplify:
        return sentences
    
    # Basic simplification: split on semicolons and coordinating conjunctions
    simplified = []
    for sentence in sentences:
        # Split on semicolons
        parts = [part.strip() for part in sentence.split(';') if part.strip()]
        
        for part in parts:
            # Split on ", and", ", but", ", or" if both parts are substantial
            coord_parts = re.split(r',\s+(and|but|or)\s+', part)
            if len(coord_parts) >= 3:
                # Take first part and reconstruct second part
                first_part = coord_parts[0].strip()
                conjunction = coord_parts[1]
                second_part = coord_parts[2].strip()
                
                if len(first_part.split()) >= 4 and len(second_part.split()) >= 4:
                    if not first_part.endswith('.'):
                        first_part += '.'
                    if not second_part.endswith('.'):
                        second_part += '.'
                    simplified.extend([first_part, second_part])
                else:
                    simplified.append(part)
            else:
                simplified.append(part)
    
    return simplified

def get_model_info() -> dict:
    """Get information about the loaded spaCy model"""
    global _nlp_model, _model_name
    
    if _nlp_model is None:
        return {"status": "not_loaded", "model": None}
    
    return {
        "status": "loaded",
        "model": _model_name,
        "version": spacy.__version__,
        "components": list(_nlp_model.pipe_names)
    }

# Initialize model on import
try:
    _ensure_spacy_model()
except Exception as e:
    warnings.warn(f"Failed to initialize spaCy model on import: {e}")
