import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import nltk
import os
from pathlib import Path
import json # ADDED for saving config
from datetime import datetime # ADDED for saving config
import argparse # Added for command-line arguments
import torch.optim as optim
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable


print(f"MatchZoo version: {mz.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="ArcI Training Script")
parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file.")
parser.add_argument("--dev_file", type=str, required=True, help="Path to the development/validation data file.")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file.")
args = parser.parse_args()

# --- Script Configuration ---
TRAIN_FILE_PATH = args.train_file 
DEV_FILE_PATH = args.dev_file 
TEST_FILE_PATH = args.test_file

EMBEDDING_FILE_PATH = r"D:/SemanticSearch/embedding/glove.6B/glove.6B.300d.txt" # Note: This path likely points to 100D embeddings. You may need to update it for 300D.
EMBEDDING_DIMENSION = 300 # Changed from 100 to 300 to match notebook

# Determine the dataset name part from the path
dataset_name_part = os.path.basename(args.train_file)
# Define the base directory for this model type
model_save_dir_base = "trained_arci_model"
# Construct the full save directory path
full_save_dir = Path(model_save_dir_base) / dataset_name_part
full_save_dir.mkdir(parents=True, exist_ok=True)
print(f"[ArcI Script] All outputs will be saved in: {full_save_dir}")

MODEL_SAVE_PATH = full_save_dir / "arci_model.pt"
PREPROCESSOR_SAVE_PATH = full_save_dir / "arci_preprocessor.dill"
CONFIG_SAVE_PATH = full_save_dir / "config.json"

FIXED_LENGTH_LEFT = 10
FIXED_LENGTH_RIGHT = 100
BATCH_SIZE = 32 # Changed from 20
EPOCHS = 10

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Attempting to download...")
    nltk_data_dir = Path(os.getcwd()) / '.nltk_data'
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.append(str(nltk_data_dir))
    try:
        nltk.download('punkt', download_dir=str(nltk_data_dir))
        print(f"'punkt' tokenizer downloaded to {nltk_data_dir} or already available there.")
    except Exception as e:
        print(f"Failed to download 'punkt': {e}. Please ensure NLTK can download data or install 'punkt' manually.")
        exit(1)

# --- Helper function to safely get parameter values (copied from other scripts) ---
def safe_get_param_value(params_table, key, default_val):
    if key in params_table: # Check if key exists
        val = params_table[key] # Retrieve item, should be the value due to ParamTable.__getitem__
        # Defensive check: if somehow a Param object itself is returned by __getitem__
        if isinstance(val, mz.engine.param.Param):
            return val.value
        return val # Assumed to be the actual value
    return default_val

# --- Task Definition ---
print("Defining ranking task for ArcI...")
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss(num_neg=4, margin=0.2)) # MODIFIED: Added num_neg=4
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(f"`ranking_task` initialized with loss: {ranking_task.losses[0]} and metrics: {ranking_task.metrics}")

# --- Helper function to load triplet data from TSV and convert to pairs ---
def load_pair_from_triplet(path, delimiter='\t'): # MODIFIED: Corrected delimiter
    """Convert a TSV triplet file (q, pos, neg) into two rows with numeric labels."""
    rows = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line_content in enumerate(f, 1):
                parts = line_content.rstrip('\n').split(delimiter) # MODIFIED: Corrected rstrip
                if len(parts) == 3:
                    q, pos, neg = parts
                    rows.append({'text_left': q, 'text_right': pos, 'label': 1})
                    rows.append({'text_left': q, 'text_right': neg, 'label': 0})
                else:
                    print(f"Warning: Line {line_num} in {path} is malformed (expected 3 parts, got {len(parts)}): '{line_content.strip()}'")
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        # Return empty list or raise error as appropriate for your workflow
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        # Return empty list or raise error
    return rows

# --- Load CUSTOM Dataset ---
print("Loading CUSTOM dataset...")
train_list_of_dicts = load_pair_from_triplet(TRAIN_FILE_PATH)
dev_list_of_dicts = load_pair_from_triplet(DEV_FILE_PATH)
test_list_of_dicts = load_pair_from_triplet(TEST_FILE_PATH)

if not train_list_of_dicts:
    print("CRITICAL: Training data is empty. Exiting.")
    exit(1)
if not dev_list_of_dicts: # Check dev data as well, as per original script logic
    print("CRITICAL: Development/Validation data is empty. Exiting.")
    exit(1)

train_df = pd.DataFrame(train_list_of_dicts)
dev_df = pd.DataFrame(dev_list_of_dicts)
test_df = pd.DataFrame(test_list_of_dicts)

# Create MatchZoo DataPacks
train_pack_raw = mz.pack(train_df, task=ranking_task)
dev_pack_raw = mz.pack(dev_df, task=ranking_task) 
test_pack_raw = mz.pack(test_df, task=ranking_task) 

print(f"Train DataPack created with {len(train_pack_raw)} entries.")
print(f"Dev (Validation) DataPack created with {len(dev_pack_raw)} entries.")
print(f"Test DataPack created with {len(test_pack_raw)} entries.")
print("CUSTOM dataset loaded and transformed into DataPacks.")

# --- Preprocessing ---
print("Preprocessing data for ArcI...")
preprocessor = mz.models.ArcI.get_default_preprocessor(
    filter_mode='df',
    filter_low_freq=2
)

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)
print("Data preprocessed.")
print(f"Preprocessor context (vocab size, etc.): {preprocessor.context}")


# --- Embedding Setup (USING YOUR CUSTOM EMBEDDINGS) ---
print(f"Setting up CUSTOM GloVe embeddings from: {EMBEDDING_FILE_PATH}")
if not os.path.exists(EMBEDDING_FILE_PATH):
    print(f"ERROR: Embedding file not found: {EMBEDDING_FILE_PATH}")
    print("Please ensure the embedding file path is correct.")
    print("Using DUMMY random embeddings as a fallback.")
    term_index_for_dummy = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = np.random.rand(len(term_index_for_dummy) + 1, EMBEDDING_DIMENSION) 
else:
    custom_embedding = mz.embedding.load_from_file(EMBEDDING_FILE_PATH, mode='glove')
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = custom_embedding.build_matrix(term_index)
    print(f"Embedding matrix built from custom file. Shape: {embedding_matrix.shape}")

    print("Normalizing embedding matrix...")
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    epsilon = 1e-8 
    embedding_matrix = embedding_matrix / (l2_norm[:, np.newaxis] + epsilon)
    embedding_matrix = np.nan_to_num(embedding_matrix, nan=0.0) 
    print(f"Normalized embedding matrix shape: {embedding_matrix.shape}")

# --- Dataset and DataLoader ---
print("Creating Datasets and DataLoaders...")
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=1, # Changed from 2
    num_neg=4, # Changed from 1
    batch_size=BATCH_SIZE, # Now 32
    resample=True,
    sort=False
)

validset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    mode='point',
    batch_size=BATCH_SIZE,
    resample=False,        
    sort=False            
)

# Padding callback
padding_callback = mz.models.ArcI.get_default_padding_callback(
    fixed_length_left=FIXED_LENGTH_LEFT,
    fixed_length_right=FIXED_LENGTH_RIGHT,
    pad_word_value=0,
    pad_word_mode='pre'
)


trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset, 
    stage='dev', 
    callback=padding_callback
)
print("Datasets and DataLoaders created.")

# --- Model Setup ---
print("Setting up ArcI model...")
model = mz.models.ArcI()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['left_length'] = FIXED_LENGTH_LEFT
model.params['right_length'] = FIXED_LENGTH_RIGHT
model.params['left_filters'] = [128]
model.params['left_kernel_sizes'] = [3]
model.params['left_pool_sizes'] = [4]
model.params['right_filters'] = [128]
model.params['right_kernel_sizes'] = [3]
model.params['right_pool_sizes'] = [4]
model.params['conv_activation_func'] = 'relu'
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 100
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'relu'
model.params['dropout_rate'] = 0.3 
model.guess_and_fill_missing_params()
model.build()
print(model)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params: {trainable_params}')

# optimizer = optim.Adadelta(model.parameters()) # Changed from Adam to Adadelta, removed lr
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Using Adam as it showed improvement

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None, 
    epochs=EPOCHS,
    save_dir=str(full_save_dir), # Ensure save_dir is a string
    patience=EPOCHS,  # Changed from EPOCHS to 3
    key=ranking_task.metrics[0]  # Changed from ranking_task.metrics
)
print("Trainer setup complete.")

# --- Training ---
print(f"Starting ArcI model training for {EPOCHS} epochs...")
trainer.run()
print("Training finished.")

# --- Consolidate Artifact Saving ---
print("Preparing and saving model, preprocessor, and config...")

# 1. Save Model
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# 2. Save Preprocessor
print(f"Saving preprocessor to: {PREPROCESSOR_SAVE_PATH}")
preprocessor.save(PREPROCESSOR_SAVE_PATH)

# 3. Create and Save Config Dictionary
config_to_save = {
    "model_name": "ArcI",
    "model_class": model.__class__.__name__,
    "task_type": ranking_task.__class__.__name__,
    "loss_function_class": ranking_task.losses[0].__class__.__name__,
    "loss_params": { # Specific to RankHingeLoss
        "margin": ranking_task.losses[0]._margin if hasattr(ranking_task.losses[0], '_margin') else 1.0,
        "strict": ranking_task.losses[0]._strict if hasattr(ranking_task.losses[0], '_strict') else False
    },
    "optimizer_class": optimizer.__class__.__name__,
    "optimizer_params": optimizer.defaults,
    "model_hyperparameters_used": {
        "left_length": safe_get_param_value(model.params, 'left_length', FIXED_LENGTH_LEFT),
        "right_length": safe_get_param_value(model.params, 'right_length', FIXED_LENGTH_RIGHT),
        "left_filters": safe_get_param_value(model.params, 'left_filters', [128]),
        "left_kernel_sizes": safe_get_param_value(model.params, 'left_kernel_sizes', [3]),
        "left_pool_sizes": safe_get_param_value(model.params, 'left_pool_sizes', [4]),
        "right_filters": safe_get_param_value(model.params, 'right_filters', [128]),
        "right_kernel_sizes": safe_get_param_value(model.params, 'right_kernel_sizes', [3]),
        "right_pool_sizes": safe_get_param_value(model.params, 'right_pool_sizes', [4]),
        "conv_activation_func": safe_get_param_value(model.params, 'conv_activation_func', 'relu'),
        "mlp_num_layers": safe_get_param_value(model.params, 'mlp_num_layers', 1),
        "mlp_num_units": safe_get_param_value(model.params, 'mlp_num_units', 100),
        "mlp_num_fan_out": safe_get_param_value(model.params, 'mlp_num_fan_out', 1),
        "mlp_activation_func": safe_get_param_value(model.params, 'mlp_activation_func', 'relu'),
        "dropout_rate": safe_get_param_value(model.params, 'dropout_rate', 0.9),
        "embedding_input_dim_from_matrix_shape": embedding_matrix.shape[0] if embedding_matrix is not None else None,
        "embedding_output_dim_from_matrix_shape": embedding_matrix.shape[1] if embedding_matrix is not None else None,
    },
    "embedding_source_file": EMBEDDING_FILE_PATH if 'EMBEDDING_FILE_PATH' in globals() and EMBEDDING_FILE_PATH and os.path.exists(EMBEDDING_FILE_PATH) else "random_dummy_embeddings_or_not_specified",
    "embedding_dim_configured": EMBEDDING_DIMENSION,
    "batch_size": BATCH_SIZE,
    "epochs_configured": EPOCHS,
    "epochs_completed": trainer._epoch if hasattr(trainer, '_epoch') else 0,
    "patience_used": trainer._early_stopping.patience if hasattr(trainer, '_early_stopping') and hasattr(trainer._early_stopping, 'patience') else None, # ArcI trainer might use default patience
    "validate_interval": trainer._validate_interval if hasattr(trainer, '_validate_interval') else None,
    "early_stopping_key": trainer._early_stopping.key if hasattr(trainer, '_early_stopping') and hasattr(trainer._early_stopping, 'key') else None,
    "device_used_for_training": str(trainer._device) if hasattr(trainer, '_device') else None,
    "trainer_verbose_level": trainer._verbose if hasattr(trainer, '_verbose') else 1,
    "preprocessor_context": {
        "fixed_length_left": preprocessor.context.get('fixed_length_left'),
        "fixed_length_right": preprocessor.context.get('fixed_length_right'),
        "filter_mode": preprocessor.context.get('filter_unit')._state.get('mode') if preprocessor.context.get('filter_unit') and hasattr(preprocessor.context.get('filter_unit'), '_state') else None,
        "filter_low_freq": preprocessor.context.get('filter_unit')._state.get('low_freq') if preprocessor.context.get('filter_unit') and hasattr(preprocessor.context.get('filter_unit'), '_state') else None,
        "vocab_size": preprocessor.context.get('vocab_size'),
        "embedding_input_dim": preprocessor.context.get('embedding_input_dim'), 
        "vocab_path": None, # Initialize to None
    },
    "matchzoo_version": mz.__version__,
    "pytorch_version": torch.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "training_script": os.path.basename(__file__),
    "training_script_path": str(Path(__file__).resolve()),
    "timestamp": datetime.now().isoformat()
}

print(f"Saving config to: {CONFIG_SAVE_PATH}")
with open(CONFIG_SAVE_PATH, 'w') as f:
    json.dump(config_to_save, f, indent=4)

print("All artifacts (model, preprocessor, config) saved.")

# --- Evaluation (Optional) ---
# ... existing evaluation code if any ...

