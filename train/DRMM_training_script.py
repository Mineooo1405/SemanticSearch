import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import nltk
import os
import sys # ADDED
from pathlib import Path
import json # ADDED for saving config
import argparse # Added for command-line arguments
# from Tool.transform_data import transform_to_matchzoo_format # Moved down
from datetime import datetime # ADDED for saving config

# --- Add project root to sys.path ---
# This ensures that modules in the project root (like 'Tool') can be found.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path modification ---

from Tool.transform_data import transform_to_matchzoo_format # MOVED HERE

print(f"MatchZoo version: {mz.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="DRMM Training Script")
parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file.")
parser.add_argument("--dev_file", type=str, required=True, help="Path to the development/validation data file.")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file.")
args = parser.parse_args()

# Determine the dataset name part from the path
dataset_name_part = os.path.basename(args.train_file)
# Define the base directory for this model type
model_save_dir_base = "trained_drmm_model"
# Construct the full save directory path using pathlib for robustness
full_save_dir = Path(model_save_dir_base) / dataset_name_part
full_save_dir.mkdir(parents=True, exist_ok=True)
print(f"[DRMM Script] All outputs will be saved in: {full_save_dir}")

EMBEDDING_FILE_PATH = r"D:/SemanticSearch/embedding/glove.6B/glove.6B.300d.txt"
EMBEDDING_DIMENSION = 300

MODEL_SAVE_PATH = full_save_dir / "drmm_model.pt"
PREPROCESSOR_SAVE_PATH = full_save_dir / "drmm_preprocessor.dill"
CONFIG_SAVE_PATH = full_save_dir / "drmm_config.json"

HIST_BIN_SIZE = 15
BATCH_SIZE = 32 # MODIFIED from 20 to 32
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

# --- Helper function to safely get parameter values (copied from MatchLSTM script) ---
def safe_get_param_value(params_table, key, default_val):
    if key in params_table: # Check if key exists
        val = params_table[key] # Retrieve item, should be the value due to ParamTable.__getitem__
        # Defensive check: if somehow a Param object itself is returned by __getitem__
        if isinstance(val, mz.engine.param.Param):
            return val.value
        return val # Assumed to be the actual value
    return default_val

print("Defining ranking task for DRMM...")
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(f"`ranking_task` initialized with loss: {ranking_task.losses[0]} and metrics: {ranking_task.metrics}")

# --- Helper function to load triplet data from TSV and convert to pairs ---
def load_pair_from_triplet(file_path, delimiter='\t'):
    """Convert a TSV triplet file (q, pos, neg) into two rows with numeric labels."""
    rows = []
    print(f"Loading and converting triplet data from: {file_path} with delimiter \'{delimiter}\'")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split(delimiter)
                if len(parts) == 3:
                    q, pos, neg = parts
                    rows.append({'text_left': q, 'text_right': pos, 'label': 1})
                    rows.append({'text_left': q, 'text_right': neg, 'label': 0})
                else:
                    print(f"Skipping malformed line #{i+1} (expected 3 columns, got {len(parts)}): {line.strip()}")
        print(f"Loaded and converted {len(rows)//2} triplets (resulting in {len(rows)} pairs) from {file_path}.")
        if not rows:
            print(f"WARNING: No data loaded/converted from {file_path}. Check the file format and delimiter.")
        return rows
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}: {e}")
        return []

TRAIN_FILE_PATH = args.train_file
DEV_FILE_PATH = args.dev_file
TEST_FILE_PATH = args.test_file

print("Loading CUSTOM dataset...")
source_train_data = load_pair_from_triplet(TRAIN_FILE_PATH)
source_dev_data = load_pair_from_triplet(DEV_FILE_PATH)
source_test_data = load_pair_from_triplet(TEST_FILE_PATH)

if not source_train_data:
    print("CRITICAL: Training data is empty. Exiting.")
    sys.exit(1) # Ensure sys is imported
if not source_dev_data:
    print("CRITICAL: Development/Validation data is empty. Exiting.")
    sys.exit(1) # Ensure sys is imported

# Create DataFrame from the loaded list of dicts
train_df = pd.DataFrame(source_train_data)
dev_df = pd.DataFrame(source_dev_data)
test_df = pd.DataFrame(source_test_data)

# Check if DataFrames are empty after loading
if train_df.empty:
    print("CRITICAL: Training DataFrame is empty after loading. Exiting.")
    sys.exit(1) # Ensure sys is imported
if dev_df.empty:
    print("CRITICAL: Development/Validation DataFrame is empty after loading. Exiting.")
    sys.exit(1) # Ensure sys is imported
if test_df.empty:
    print("WARNING: Test DataFrame is empty after loading.")

train_pack_raw = mz.pack(train_df, task=ranking_task)
dev_pack_raw = mz.pack(dev_df, task=ranking_task)
test_pack_raw = mz.pack(test_df, task=ranking_task)

print(f"Train DataPack created with {len(train_pack_raw)} entries.")
print(f"Dev (Validation) DataPack created with {len(dev_pack_raw)} entries.")
print(f"Test DataPack created with {len(test_pack_raw)} entries.")

print("Preprocessing data for DRMM...")
preprocessor = mz.models.DRMM.get_default_preprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)
print("Data preprocessed.")
print(f"Preprocessor context (vocab size, etc.): {preprocessor.context}")

print(f"Setting up CUSTOM GloVe embeddings from: {EMBEDDING_FILE_PATH}")
if not os.path.exists(EMBEDDING_FILE_PATH):
    print(f"ERROR: Embedding file not found: {EMBEDDING_FILE_PATH}")
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
    epsilon = 1e-8 # To prevent division by zero for zero vectors
    embedding_matrix = embedding_matrix / (l2_norm[:, np.newaxis] + epsilon)
    embedding_matrix = np.nan_to_num(embedding_matrix, nan=0.0)
    print(f"Normalized embedding matrix shape: {embedding_matrix.shape}")

print("Setting up callbacks...")
histgram_callback = mz.dataloader.callbacks.Histogram(
    embedding_matrix, bin_size=HIST_BIN_SIZE, hist_mode='LCH'
)
padding_callback = mz.models.DRMM.get_default_padding_callback()
print("Callbacks created.")

print("Creating Datasets and DataLoaders...")
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=1,
    num_neg=4,
    batch_size=BATCH_SIZE,
    resample=True,
    sort=False,
    callbacks=[histgram_callback]
)
validset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    batch_size=BATCH_SIZE,
    resample=False,
    sort=False,
    callbacks=[histgram_callback]
)

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback,
    device='cpu' # Added to match notebook
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    stage='dev',
    callback=padding_callback,
    device='cpu' # Added to match notebook
)
print("Datasets and DataLoaders created.")

print("Setting up DRMM model...")
model = mz.models.DRMM()
model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['embedding_freeze'] = True # Ensure embeddings are trainable
model.params['mask_value'] = 0
model.params['hist_bin_size'] = HIST_BIN_SIZE
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 1
model.params['mlp_activation_func'] = 'tanh'

model.build()
print("DRMM Model built.")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print("Setting up Trainer...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # CHANGED from Adadelta
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3) # Optional: add if needed

NUM_TRAIN_EPOCHS = EPOCHS # Ensure this is defined, e.g., 10 or from args # CHANGED from 10 to EPOCHS

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None, 
    epochs=NUM_TRAIN_EPOCHS, # Use defined epochs
    # scheduler=scheduler, # Optional
    clip_norm=10, # Optional: add if needed
    patience=NUM_TRAIN_EPOCHS, # Added patience for early stopping if validloader is used
    key=ranking_task.metrics[0], # Monitor the first metric for early stopping
    device='cpu' # Added to match notebook
)
print("Trainer setup complete.")

print(f"Starting DRMM model training for {EPOCHS} epochs...")
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
    "model_name": "DRMM",
    "model_class": model.__class__.__name__,
    "task_type": ranking_task.__class__.__name__, # Used ranking_task directly
    "loss_function_class": ranking_task.losses[0].__class__.__name__, # MODIFIED: Access the first loss in the list
    "loss_num_neg": ranking_task.losses[0]._num_neg if hasattr(ranking_task.losses[0], '_num_neg') else None, # MODIFIED: Directly access attribute
    "optimizer_class": optimizer.__class__.__name__,
    "optimizer_params": optimizer.defaults,
    "model_hyperparameters_used": {
        "mask_value": safe_get_param_value(model.params, 'mask_value', 0),
        "hist_bin_size": safe_get_param_value(model.params, 'hist_bin_size', HIST_BIN_SIZE),
        "mlp_num_layers": safe_get_param_value(model.params, 'mlp_num_layers', 1),
        "mlp_num_units": safe_get_param_value(model.params, 'mlp_num_units', 10),
        "mlp_num_fan_out": safe_get_param_value(model.params, 'mlp_num_fan_out', 1),
        "mlp_activation_func": safe_get_param_value(model.params, 'mlp_activation_func', 'tanh'),
        # embedding_matrix is directly passed, so its dimensions are key
        "embedding_input_dim_from_matrix_shape": embedding_matrix.shape[0] if embedding_matrix is not None else None,
        "embedding_output_dim_from_matrix_shape": embedding_matrix.shape[1] if embedding_matrix is not None else None,
    },
    "embedding_source_file": EMBEDDING_FILE_PATH if 'EMBEDDING_FILE_PATH' in globals() and EMBEDDING_FILE_PATH and os.path.exists(EMBEDDING_FILE_PATH) else "random_dummy_embeddings_or_not_specified",
    "embedding_dim_configured": EMBEDDING_DIMENSION,
    "batch_size_configured": BATCH_SIZE,
    "epochs_configured": EPOCHS,
    "hist_bin_size_configured": HIST_BIN_SIZE,
    "num_neg_train_dataset": trainset.num_neg if hasattr(trainset, 'num_neg') else None,
    "num_dup_train_dataset": trainset.num_dup if hasattr(trainset, 'num_dup') else None,
    "fixed_length_left_preprocessor": preprocessor.context.get('fixed_length_left'),
    "fixed_length_right_preprocessor": preprocessor.context.get('fixed_length_right'),
    "vocab_size_from_preprocessor_context": preprocessor.context.get('vocab_size'),
    "matchzoo_version": mz.__version__,
    "training_script_path": str(Path(__file__).resolve()),
    "timestamp": datetime.now().isoformat()
}

print(f"Saving config to: {CONFIG_SAVE_PATH}")
with open(CONFIG_SAVE_PATH, 'w') as f:
    json.dump(config_to_save, f, indent=4)

print("All artifacts (model, preprocessor, config) saved.")

# --- Evaluation (Optional) ---
# ...existing code...
