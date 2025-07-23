import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import importlib # ADDED
from typing import Optional

try:
    torch_directml = importlib.import_module("torch_directml")
    if not (hasattr(torch_directml, 'is_available') and callable(torch_directml.is_available) and
            hasattr(torch_directml, 'device') and callable(torch_directml.device)):
        torch_directml = None
except ImportError:
    torch_directml = None
except Exception:
    torch_directml = None


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global dictionary to cache models
loaded_models = {}
_embedding_cache = {}

def get_device(preferred_device_str: Optional[str] = None):
    """Determines the appropriate torch device."""
    if preferred_device_str:
        if preferred_device_str == 'dml':
            if torch_directml and hasattr(torch_directml, 'is_available') and torch_directml.is_available() and hasattr(torch_directml, 'device'):
                try:
                    dml_device = torch_directml.device()
                    return dml_device
                except Exception:
                    pass
        elif preferred_device_str == 'cuda' and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred_device_str == 'cpu':
            return torch.device("cpu")

    # Auto-detection if no valid preference
    if torch_directml and hasattr(torch_directml, 'is_available') and torch_directml.is_available() and hasattr(torch_directml, 'device'):
        try:
            dml_device = torch_directml.device()
            return dml_device
        except Exception:
            pass
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")

def sentence_embedding(text_list: list, model_name: str, batch_size: int = 32, device_preference: Optional[str] = None):
    """Embeds a list of texts using a specified SentenceTransformer model."""
    global loaded_models

    effective_device = get_device(device_preference)
    
    # Special handling for DirectML: try DirectML first, fallback to CPU if error
    if str(effective_device) == 'privateuseone:0':  # DirectML device
        try:
            if model_name not in loaded_models:
                loaded_models[model_name] = SentenceTransformer(model_name, device=str(effective_device))
            
            model = loaded_models[model_name]
            embeddings = model.encode(text_list, batch_size=batch_size, show_progress_bar=False)
            return embeddings
            
        except Exception as directml_error:
            logging.warning(f"DirectML failed for model {model_name}: {directml_error}")
            logging.info("Falling back to CPU for embeddings...")
            # Fallback to CPU
            effective_device = torch.device('cpu')
            if model_name in loaded_models:
                del loaded_models[model_name]  # Remove the failed DirectML model
    
    # Standard loading for CPU/CUDA
    if model_name not in loaded_models:
        try:
            loaded_models[model_name] = SentenceTransformer(model_name, device=str(effective_device))
        except Exception as e:
            if str(effective_device) != 'cpu':
                try:
                    cpu_device = torch.device('cpu')
                    loaded_models[model_name] = SentenceTransformer(model_name, device=str(cpu_device))
                    effective_device = cpu_device
                except Exception:
                    raise
            else:
                raise
    else:
        current_model_device = loaded_models[model_name].device
        if str(current_model_device) != str(effective_device):
            try:
                loaded_models[model_name].to(effective_device)
            except Exception:
                effective_device = current_model_device

    model = loaded_models[model_name]
    
    try:
        embeddings = model.encode(text_list, batch_size=batch_size, show_progress_bar=False)
    except Exception as e:
        logging.error(f"Error during sentence encoding with model {model_name}: {e}")
        raise

    return embeddings

if __name__ == '__main__':
    # Example usage
    texts = ["This is a test sentence.", "Another example sentence."]
    try:
        embeddings = sentence_embedding(texts, 'all-MiniLM-L6-v2', batch_size=2)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Model device: {loaded_models['all-MiniLM-L6-v2'].device}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("Example script finished.")