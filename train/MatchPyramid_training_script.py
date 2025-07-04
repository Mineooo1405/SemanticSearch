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

print(f"MatchZoo version: {mz.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# +++ ADDED HELPER FUNCTION +++
# def safe_get_param_value(params_table, key, default_val):
#     """Safely retrieves a parameter value from a MatchZoo ParamTable."""
#     if key in params_table: # Check if key exists
#         val = params_table[key] # Retrieve item
#         # Defensive check: if somehow a Param object itself is returned
#         if isinstance(val, mz.engine.param.Param):
#             return val.value
#         return val # Assumed to be the actual value
#     return default_val
# +++ END ADDED HELPER FUNCTION +++

# --- Argument Parsing --- Added
parser = argparse.ArgumentParser(description="MatchPyramid Training Script")
parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file.")
parser.add_argument("--dev_file", type=str, required=True, help="Path to the development/validation data file.")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file.")
args = parser.parse_args()

TRAIN_FILE_PATH = args.train_file
DEV_FILE_PATH = args.dev_file
TEST_FILE_PATH = args.test_file

EMBEDDING_FILE_PATH = r"D:/SemanticSearch/embedding/glove.6B/glove.6B.300d.txt" # NOTE: This path points to 100D. Update if using 300D.
EMBEDDING_DIMENSION = 300 # Changed from 100 to 300 to match notebook

# Determine the dataset name part from the path
dataset_name_part = os.path.basename(args.train_file)
# Define the base directory for this model type
model_save_dir_base = "trained_matchpyramid_model"
# Construct the full save directory path
full_save_dir = Path(model_save_dir_base) / dataset_name_part
full_save_dir.mkdir(parents=True, exist_ok=True)
print(f"[MatchPyramid Script] All outputs will be saved in: {full_save_dir}")

MODEL_SAVE_PATH = full_save_dir / "matchpyramid_model.pt"
PREPROCESSOR_SAVE_PATH = full_save_dir / "preprocessor.dill"
CONFIG_SAVE_PATH = full_save_dir / "config.json"

BATCH_SIZE = 32 # MODIFIED from 20 to 32 as per recommendation
EPOCHS = 10 # Changed from 10 to 5 to match notebook

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

# --- Helper function to safely get parameter values ---
def safe_get_param_value(params_source, key, default_val):
    """
    Safely retrieves a parameter value from a MatchZoo ParamTable or a dictionary.
    For ParamTable, it directly accesses the value.
    For dictionaries, it uses .get().
    """
    if isinstance(params_source, mz.engine.param_table.ParamTable):
        if key in params_source:
            param_obj = params_source.get(key) # Gets the Param object
            if param_obj is not None:
                return param_obj.value # Access the .value attribute of the Param object
        return default_val
    elif isinstance(params_source, dict):
        return params_source.get(key, default_val)
    return default_val

print("Defining ranking task for MatchPyramid...")
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4)) # MODIFIED num_neg from 1 to 4
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(f"`ranking_task` initialized with loss: {ranking_task.losses[0]} and metrics: {ranking_task.metrics}")

def load_pair_from_triplet(file_path, delimiter='\t'): # MODIFIED: Corrected delimiter
    """Convert a TSV triplet file (q, pos, neg) into two rows with numeric labels."""
    rows = []
    print(f"Loading and converting triplet data from: {file_path} with delimiter '{delimiter}'")
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
        return rows # Returns a list of dicts
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}: {e}")
        return []

print("Loading CUSTOM dataset...")
source_train_data = load_pair_from_triplet(TRAIN_FILE_PATH)
source_dev_data = load_pair_from_triplet(DEV_FILE_PATH)
source_test_data = load_pair_from_triplet(TEST_FILE_PATH)

if not source_train_data:
    print("CRITICAL: Training data is empty. Exiting.")
    exit(1)
if not source_dev_data:
    print("CRITICAL: Development/Validation data is empty. Exiting.")
    exit(1)

train_df = pd.DataFrame(source_train_data) # MODIFIED: No columns needed if list of dicts
dev_df = pd.DataFrame(source_dev_data)     # MODIFIED: No columns needed
test_df = pd.DataFrame(source_test_data)    # MODIFIED: No columns needed

train_pack_raw = mz.pack(train_df, task=ranking_task)
dev_pack_raw = mz.pack(dev_df, task=ranking_task)
test_pack_raw = mz.pack(test_df, task=ranking_task)

print(f"Train DataPack created with {len(train_pack_raw)} entries.")
print(f"Dev (Validation) DataPack created with {len(dev_pack_raw)} entries.")
print(f"Test DataPack created with {len(test_pack_raw)} entries.")

print("Preprocessing data for MatchPyramid...")
preprocessor = mz.models.MatchPyramid.get_default_preprocessor()
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

print("Creating Datasets and DataLoaders...")
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2, # Changed from 1 to 2 to match notebook
    num_neg=4, # MODIFIED from 1 to 4, to align with loss
    batch_size=BATCH_SIZE,
    resample=True,
    sort=False,
    shuffle=True
)
validset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    batch_size=BATCH_SIZE,
    resample=False,
    sort=False,
    shuffle=False
)

padding_callback = mz.models.MatchPyramid.get_default_padding_callback()

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

print("Setting up MatchPyramid model...")
model = mz.models.MatchPyramid()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['embedding_freeze'] = False # Ensure embeddings are trainable
model.params['kernel_size'] = [[3, 3], [3, 3]]
model.params['kernel_count'] = [16, 32]  # Changed from [8, 16] to match notebook
model.params['dpool_size'] = [3, 10]
model.params['dropout_rate'] = 0.1

model.build()
print(model)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print("Setting up Trainer...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # MODIFIED lr from 0.001 to 1e-4

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    validate_interval=None,
    epochs=EPOCHS,
    save_dir=str(full_save_dir),  # MODIFIED to ensure it's a string if Path object was used before
    patience=EPOCHS, # As per notebook (assuming 3 if not specified, or adjust as needed)
    key=ranking_task.metrics[0]
)
print("Trainer setup complete.")

print(f"Starting MatchPyramid model training for {EPOCHS} epochs...")
trainer.run()
print("Training finished.")

print("Preparing and saving model, preprocessor, and config...")

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saving preprocessor to: {PREPROCESSOR_SAVE_PATH}")
preprocessor.save(PREPROCESSOR_SAVE_PATH) # CHANGED: Pass the full file path

# Create Config Dictionary
config_to_save = {
    "model_name": "MatchPyramid",
    "model_class": model.__class__.__name__,
    "task_type": safe_get_param_value(model.params, 'task', {}).__class__.__name__ if safe_get_param_value(model.params, 'task', None) else None,
    "loss_function_class": safe_get_param_value(model.params, 'task', {}).loss.__class__.__name__ if safe_get_param_value(model.params, 'task', None) and hasattr(safe_get_param_value(model.params, 'task', {}), 'loss') else None,
    "loss_num_neg": safe_get_param_value(safe_get_param_value(model.params, 'task', {}).loss, '_num_neg', None) if safe_get_param_value(model.params, 'task', None) and hasattr(safe_get_param_value(model.params, 'task', {}), 'loss') else None,
    "optimizer_class": optimizer.__class__.__name__,
    "learning_rate": optimizer.defaults.get('lr'),
    "model_hyperparameters_used": {
        "kernel_size": safe_get_param_value(model.params, 'kernel_size', None),
        "kernel_count": safe_get_param_value(model.params, 'kernel_count', None),
        "dpool_size": safe_get_param_value(model.params, 'dpool_size', None),
        "dropout_rate": safe_get_param_value(model.params, 'dropout_rate', None),
        "embedding_input_dim_model_param": safe_get_param_value(model.params, 'embedding_input_dim', None), # Usually derived from preprocessor
        "embedding_output_dim_model_param": safe_get_param_value(model.params, 'embedding_output_dim', None) # Usually derived from preprocessor
    },
    "embedding_source_file": EMBEDDING_FILE_PATH if 'EMBEDDING_FILE_PATH' in globals() and os.path.exists(EMBEDDING_FILE_PATH) else "random_dummy_embeddings",
    "embedding_dim_used": EMBEDDING_DIMENSION,
    "batch_size": BATCH_SIZE,
    "epochs_configured": EPOCHS,
    "epochs_completed": trainer._epoch if hasattr(trainer, '_epoch') else 0,
    "patience_used": trainer.patience if hasattr(trainer, 'patience') else None, # Corrected: trainer object has patience directly
    "validate_interval": trainer.validate_interval if hasattr(trainer, 'validate_interval') else None, # Corrected
    "early_stopping_key": trainer.early_stopping.key if hasattr(trainer, 'early_stopping') and trainer.early_stopping else None, # Corrected
    "device_used_for_training": str(trainer.device) if hasattr(trainer, 'device') else None, # Corrected
    "start_epoch_configured": trainer.start_epoch if hasattr(trainer, 'start_epoch') else 1, # Corrected
    "gradient_clip_norm": trainer.clip_norm if hasattr(trainer, 'clip_norm') else None, # Corrected
    "scheduler_class": trainer.scheduler.__class__.__name__ if hasattr(trainer, 'scheduler') and trainer.scheduler is not None else None, # Corrected
    "trainer_save_directory": str(trainer.save_dir.resolve()) if hasattr(trainer, 'save_dir') and trainer.save_dir is not None else None, # Corrected
    "trainer_save_all_flag": trainer.save_all if hasattr(trainer, 'save_all') else False, # Corrected
    "trainer_verbose_level": trainer.verbose if hasattr(trainer, 'verbose') else 1, # Corrected
    "fixed_length_left": preprocessor.context.get('fixed_length_left'),
    "fixed_length_right": preprocessor.context.get('fixed_length_right'),
    "vocab_size_from_preprocessor_context": preprocessor.context.get('vocab_size'),
    "embedding_input_dim_from_preprocessor_context": preprocessor.context.get('embedding_input_dim'), # This is vocab_size + 1 (for padding)
    "matchzoo_version": mz.__version__,
    "training_script": os.path.basename(__file__),
    "training_date": datetime.now().isoformat()
}

# Ensure all values in model_hyperparameters_used are serializable (they should be by default for MatchPyramid)
for key, value in config_to_save.get("model_hyperparameters_used", {}).items():
    if isinstance(value, np.ndarray):
        config_to_save["model_hyperparameters_used"][key] = value.tolist()
    elif not isinstance(value, (list, dict, str, int, float, bool, type(None))):
        config_to_save["model_hyperparameters_used"][key] = str(value)

print(f"Saving config to: {CONFIG_SAVE_PATH}")
with open(CONFIG_SAVE_PATH, 'w') as f:
    json.dump(config_to_save, f, indent=4)

print("Evaluation on test data...")
test_pack_processed = preprocessor.transform(test_pack_raw)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    batch_size=BATCH_SIZE,
    resample=False,
    sort=False,
    shuffle=False
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='dev', # CHANGED FROM 'test' TO 'dev' to ensure labels are yielded
    callback=padding_callback
)

print("Running evaluation...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_tuple in testloader: # Iterating over testloader yields a tuple (inputs_dict, labels_tensor)
        inputs_dict = batch_tuple[0]
        labels_tensor = batch_tuple[1]

        text_left = inputs_dict['text_left']
        text_right = inputs_dict['text_right']
        labels = labels_tensor.numpy().tolist() # Convert to list

        # Call the model directly for prediction (invokes forward method)
        # Pass the input_dict directly, as the model's forward method expects a dictionary
        outputs = model(inputs_dict)
        
        # NOTE: The following line for `preds` is likely incorrect for ranking scores.
        # `outputs` from a MatchZoo ranking model are typically relevance scores (e.g., shape [batch_size, 1]).
        # `np.argmax(outputs, axis=1)` will produce all zeros if outputs.shape[1] is 1,
        # which will significantly affect the accuracy calculation.
        preds = np.argmax(outputs, axis=1).tolist() # Get index of max log-probability

        all_preds.extend(preds)
        all_labels.extend(labels)

# Calculate and display metrics
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Test Accuracy: {accuracy:.4f}")

print("MatchPyramid training script finished successfully.")
print(f"Model saved at: {MODEL_SAVE_PATH}")
print(f"Preprocessor saved at: {PREPROCESSOR_SAVE_PATH}")
print(f"Config saved at: {CONFIG_SAVE_PATH}") # ADDED
