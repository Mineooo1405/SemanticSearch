'''
Script to split the main training data into a new, smaller training set and a development set.
'''
import pandas as pd
from sklearn.model_selection import train_test_split
import os # Added import

# Define file paths
# USER ACTION: Modify this path to point to your specific training data file.
main_train_file_path = "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_train_mz.tsv"

# --- Derive output paths based on main_train_file_path ---
_input_dir = os.path.dirname(main_train_file_path)
_input_filename = os.path.basename(main_train_file_path)
_file_stem, _file_ext = os.path.splitext(_input_filename)

_base_name_for_outputs = _file_stem
# Try to strip known training suffixes to get a cleaner base name for outputs
# Order is important: from most specific to least specific
if _file_stem.endswith("_train_train_mz"):
    _base_name_for_outputs = _file_stem[:-len("_train_train_mz")]
elif _file_stem.endswith("_train_mz"):
    _base_name_for_outputs = _file_stem[:-len("_train_mz")]
# Ensure we don't strip "_train" if the filename is just "train"
elif _file_stem.endswith("_train") and _file_stem != "train":
    _base_name_for_outputs = _file_stem[:-len("_train")]

# If stripping resulted in an empty string (e.g. input was "_train.tsv"), revert to original stem
if not _base_name_for_outputs and _file_stem:
    _base_name_for_outputs = _file_stem

# Define output file paths using the derived base name and input directory
output_train_file_path = os.path.join(_input_dir, f"{_base_name_for_outputs}_train{_file_ext}")
output_dev_file_path = os.path.join(_input_dir, f"{_base_name_for_outputs}_test{_file_ext}")
# --- End Derivation of output paths ---

# Define the split ratio
# 0.1 means 10% for the development set, 90% for the new training set
dev_set_size = 0.1
random_seed = 42 # for reproducibility

print(f"Input training data: {main_train_file_path}") # Modified print statement
print(f"Output new training file: {output_train_file_path}") # Added print statement
print(f"Output new development file: {output_dev_file_path}") # Added print statement

# Read the main training data
# Assuming the file has no header and is tab-separated.
# For MatchZoo, typical columns might be label, id_left, text_left, id_right, text_right
# Or for triplet loss: query, positive_passage, negative_passage
# The script currently assumes 3 columns if no names are given, which might be too specific.
# For more general use, it might be better to not specify names or handle varying column counts.
try:
    # Reading without 'names' makes it more generic for typical MatchZoo .tsv files
    main_df = pd.read_csv(main_train_file_path, sep='\t', header=None)
except FileNotFoundError:
    print(f"Error: The file {main_train_file_path} was not found.")
    exit()
except Exception as e:
    print(f"Error reading {main_train_file_path}: {e}")
    exit()

print(f"Splitting data: {1-dev_set_size:.0%} for new training, {dev_set_size:.0%} for development.")
# Split the data
# The original train_df from dataset_split.py is now being split further.
new_train_df, new_dev_df = train_test_split(
    main_df, 
    test_size=dev_set_size, 
    random_state=random_seed
)

print(f"Saving new training set to: {output_train_file_path}")
# Save the new training set
new_train_df.to_csv(output_train_file_path, sep='\t', index=False, header=False)

print(f"Saving new development set to: {output_dev_file_path}")
# Save the new development set
new_dev_df.to_csv(output_dev_file_path, sep='\t', index=False, header=False)

print("Script finished successfully.")
print(f"New training set shape: {new_train_df.shape}")
print(f"New development set shape: {new_dev_df.shape}")
