import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import importlib # ADDED
from typing import Optional

# Attempt to import torch_directml and make it robust
# This is primarily for Windows with DirectML. On other systems, it will gracefully fallback.
try:
    torch_directml = importlib.import_module("torch_directml")
    if not (hasattr(torch_directml, 'is_available') and callable(torch_directml.is_available) and
            hasattr(torch_directml, 'device') and callable(torch_directml.device)):
        # print("Warning (Sentence_Embedding.py): torch_directml imported but essential attributes are missing. Treating as unavailable.")
        logging.warning("Sentence_Embedding.py: torch_directml imported but essential attributes (is_available, device) are missing or not callable. Treating as unavailable.")
        torch_directml = None # Invalidate if key functions are missing
except ImportError:
    # print("Info (Sentence_Embedding.py): torch_directml not found. Will use CPU or CUDA if available.")
    logging.info("Sentence_Embedding.py: torch_directml not found. PyTorch will use CPU or CUDA if available.")
    torch_directml = None
except Exception as e_tdml_import: # Catch any other exception during import, including OSError
    # print(f"Warning (Sentence_Embedding.py): An unexpected error occurred during torch_directml import: {e_tdml_import}. Treating as unavailable.")
    logging.warning(f"Sentence_Embedding.py: An unexpected error occurred during torch_directml import: {e_tdml_import}. Treating as unavailable.")
    torch_directml = None


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global dictionary to cache models
# Key: model_name, Value: loaded SentenceTransformer model
loaded_models = {}
# Cache for individual text embeddings per model: {(model_name, text): np.ndarray}
_embedding_cache = {}

def get_device(preferred_device_str: Optional[str] = None):
    """
    Determines the appropriate torch device.
    Prioritizes DirectML if available and functional, then CUDA, then CPU.
    A preferred_device_str (e.g., 'dml', 'cuda', 'cpu') can override auto-detection.
    """
    if preferred_device_str:
        if preferred_device_str == 'dml':
            if torch_directml and hasattr(torch_directml, 'is_available') and torch_directml.is_available() and hasattr(torch_directml, 'device'):
                try:
                    # Check if the device can be created
                    dml_device = torch_directml.device()
                    # Basic test: create a tensor on the device
                    # _ = torch.tensor([1.0], device=dml_device) # Commented out to avoid issues if model loading fails later
                    logging.info(f"Using preferred DirectML device: {dml_device}")
                    return dml_device
                except Exception as e:
                    logging.warning(f"Preferred DirectML device specified but failed to initialize/test: {e}. Falling back.")
            else:
                logging.warning("Preferred DirectML device specified but torch_directml is not available/functional. Falling back.")
        elif preferred_device_str == 'cuda' and torch.cuda.is_available():
            logging.info("Using preferred CUDA device.")
            return torch.device("cuda")
        elif preferred_device_str == 'cpu':
            logging.info("Using preferred CPU device.")
            return torch.device("cpu")
        else:
            logging.warning(f"Preferred device '{preferred_device_str}' not recognized or not available. Auto-detecting.")

    # Auto-detection if no valid preference
    if torch_directml and hasattr(torch_directml, 'is_available') and torch_directml.is_available() and hasattr(torch_directml, 'device'):
        try:
            dml_device = torch_directml.device()
            # _ = torch.tensor([1.0], device=dml_device) # Basic test
            logging.info(f"Using DirectML device: {dml_device}")
            return dml_device
        except Exception as e:
            logging.warning(f"DirectML available but failed to initialize/test: {e}. Falling back to CUDA/CPU.")
    
    if torch.cuda.is_available():
        logging.info("Using CUDA device.")
        return torch.device("cuda")
    
    logging.info("Using CPU device.")
    return torch.device("cpu")

def sentence_embedding(text_list: list, model_name: str, batch_size: int = 32, device_preference: Optional[str] = None):
    """
    Embeds a list of texts using a specified SentenceTransformer model.
    Uses a global cache for loaded models to avoid reloading.
    Allows specifying a device preference ('dml', 'cuda', 'cpu').
    """
    global loaded_models

    # Determine the device to use
    effective_device = get_device(device_preference)
    # logging.info(f"Sentence embedding will run on device: {effective_device}")


    if model_name not in loaded_models:
        logging.info(f"Loading SentenceTransformer model: {model_name} onto device: {effective_device}")
        try:
            loaded_models[model_name] = SentenceTransformer(model_name, device=str(effective_device)) # Pass device as string
            logging.info(f"Model {model_name} loaded successfully on {effective_device}.")
        except Exception as e:
            logging.error(f"Failed to load model {model_name} on device {effective_device}: {e}")
            # Fallback to CPU if model loading failed on the preferred device (unless it was already CPU)
            if str(effective_device) != 'cpu':
                logging.warning(f"Attempting to load model {model_name} on CPU as a fallback.")
                try:
                    cpu_device = torch.device('cpu')
                    loaded_models[model_name] = SentenceTransformer(model_name, device=str(cpu_device))
                    logging.info(f"Model {model_name} loaded successfully on CPU (fallback).")
                    effective_device = cpu_device # Update effective_device if fallback was successful
                except Exception as e_cpu:
                    logging.error(f"Failed to load model {model_name} on CPU (fallback): {e_cpu}")
                    raise # Re-raise the exception if CPU fallback also fails
            else:
                raise # Re-raise if it failed on CPU initially
    else:
        # If model exists, ensure it's on the currently desired effective_device
        # This handles cases where the device_preference might change between calls for the same model
        # Or if the model was loaded on CPU due to a previous DML/CUDA failure.
        current_model_device = loaded_models[model_name].device
        if str(current_model_device) != str(effective_device):
            logging.info(f"Model {model_name} is on {current_model_device}, but requested device is {effective_device}. Moving model.")
            try:
                loaded_models[model_name].to(effective_device)
                logging.info(f"Model {model_name} moved to {effective_device}.")
            except Exception as e_move:
                logging.error(f"Failed to move model {model_name} to {effective_device}: {e_move}. Using current device {current_model_device}.")
                # If moving fails, we might have to use the model on its current device or attempt a CPU fallback.
                # For simplicity here, we'll log the error and proceed with the model on its current device.
                # A more robust solution might try to reload on CPU if the move fails and current device is not CPU.
                effective_device = current_model_device # Use the device the model is actually on


    model = loaded_models[model_name]
    
    # --- NOTE ---
    # Đã loại bỏ cơ chế cache embeddings trong RAM. Luôn tính mới mỗi lần gọi.

    try:
        embeddings = model.encode(text_list, batch_size=batch_size, show_progress_bar=False)
    except Exception as e:
        logging.error(f"Error during sentence encoding with model {model_name}: {e}")
        raise

    return embeddings

if __name__ == '__main__':
    # Example Usage:
    # Pass device_preference to test specific devices if needed.
    # e.g., device_preference='dml', device_preference='cuda', device_preference='cpu'
    # If None, it will auto-detect.

    # Determine preferred device for the example from environment or default
    # This is just for the example, your main script might pass this differently
    preferred_device_for_example = None # or 'dml', 'cuda', 'cpu'
    
    print(f"--- Testing with device_preference: {preferred_device_for_example} ---")

    # Test 1: A common small model
    try:
        print("\\nTesting with 'all-MiniLM-L6-v2'...")
        texts1 = ["This is a test sentence.", "Another example sentence."]
        embeddings1 = sentence_embedding(texts1, 'all-MiniLM-L6-v2', batch_size=2, device_preference=preferred_device_for_example)
        print(f"Embeddings for 'all-MiniLM-L6-v2' (shape: {embeddings1.shape}):\\n{embeddings1}")
        print(f"Model 'all-MiniLM-L6-v2' is on device: {loaded_models['all-MiniLM-L6-v2'].device}")
    except Exception as e:
        print(f"Error testing 'all-MiniLM-L6-v2': {e}")

    # Test 2: A larger model that might be more sensitive to device memory
    try:
        print("\\nTesting with 'thenlper/gte-large' (might take a moment to download)...")
        texts2 = ["Exploring the capabilities of large language models.", 
                  "Semantic search and information retrieval."]
        embeddings2 = sentence_embedding(texts2, 'thenlper/gte-large', batch_size=1, device_preference=preferred_device_for_example)
        print(f"Embeddings for 'thenlper/gte-large' (shape: {embeddings2.shape}):\\n{embeddings2}")
        print(f"Model 'thenlper/gte-large' is on device: {loaded_models['thenlper/gte-large'].device}")
    except Exception as e:
        print(f"Error testing 'thenlper/gte-large': {e}")

    # Test 3: Using the same model again (should use cache and potentially different device if preference changes)
    try:
        print("\\nTesting 'all-MiniLM-L6-v2' again (cached)...")
        # Example: Test with a different device preference for the cached model
        # preferred_device_for_example_2 = 'cpu' # or 'cuda' if different from first test
        # print(f"--- Testing cached model with device_preference: {preferred_device_for_example_2} ---")
        embeddings3 = sentence_embedding(texts1, 'all-MiniLM-L6-v2', batch_size=2, device_preference=preferred_device_for_example) # Change here if testing device switch
        print(f"Embeddings for 'all-MiniLM-L6-v2' (cached) (shape: {embeddings3.shape}):\\n{embeddings3}")
        print(f"Model 'all-MiniLM-L6-v2' is on device: {loaded_models['all-MiniLM-L6-v2'].device}")
    except Exception as e:
        print(f"Error testing 'all-MiniLM-L6-v2' (cached): {e}")

    print("\\n--- Device check after tests ---")
    final_device = get_device(preferred_device_for_example)
    print(f"Final get_device() call with preference '{preferred_device_for_example}' resolves to: {final_device}")
    final_device_auto = get_device()
    print(f"Final get_device() call with no preference resolves to: {final_device_auto}")

    print("\\nExample script finished.")