import os
import subprocess
import sys

# Define the directory containing the training scripts
TRAIN_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "train")

def get_available_training_scripts():
    """Scans the TRAIN_SCRIPTS_DIR for Python training scripts."""
    scripts = []
    if not os.path.isdir(TRAIN_SCRIPTS_DIR):
        print(f"Error: Training scripts directory not found: {TRAIN_SCRIPTS_DIR}")
        return scripts
    for fname in os.listdir(TRAIN_SCRIPTS_DIR):
        if fname.endswith("_training_script.py") and os.path.isfile(os.path.join(TRAIN_SCRIPTS_DIR, fname)):
            # Extract model name from filename (e.g., "MatchLSTM" from "MatchLSTM_training_script.py")
            model_name = fname.replace("_training_script.py", "")
            scripts.append({"name": model_name, "path": os.path.join(TRAIN_SCRIPTS_DIR, fname)})
    return scripts

def get_user_data_paths():
    """Prompts the user to enter paths for train, dev, and test datasets."""
    print("--- Dataset Configuration ---")
    while True:
        train_path = input("Enter the absolute path for the TRAINING dataset (e.g., D:/data/train.tsv or D:/data/train.tsv): ").strip()
        if os.path.isfile(train_path):
            break
        print(f"Error: File not found or path is not a file: {train_path}. Please try again.")

    while True:
        dev_path = input("Enter the absolute path for the DEVELOPMENT/VALIDATION dataset (e.g., D:/data/dev.tsv or D:/data/dev.tsv): ").strip()
        if os.path.isfile(dev_path):
            break
        print(f"Error: File not found or path is not a file: {dev_path}. Please try again.")

    while True:
        test_path = input("Enter the absolute path for the TEST dataset (e.g., D:/data/test.tsv or D:/data/test.tsv): ").strip()
        if os.path.isfile(test_path):
            break
        print(f"Error: File not found or path is not a file: {test_path}. Please try again.")
    return train_path, dev_path, test_path # Corrected: test_file to test_path

def run_training_script(script_path, train_file, dev_file, test_file):
    """
    Runs the target training script by passing dataset paths as command-line arguments.
    """
    print(f"--- Training Model from: {os.path.basename(script_path)} ---")

    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    
    command = [
        sys.executable, 
        script_path,
        "--train_file", train_file,
        "--dev_file", dev_file,
        "--test_file", test_file
    ]
    
    print(f"Running script: {' '.join(command)} with cwd: {project_root_dir}...")
    try:
        process = subprocess.run(
            command,
            capture_output=False, # Set to True if you want to capture stdout/stderr
            text=True, 
            check=False, # Set to True to raise an exception for non-zero exit codes
            cwd=project_root_dir  # Set current working directory
        )

        if process.returncode == 0:
            print(f"Successfully completed training for {os.path.basename(script_path)}.")
            success = True
        else:
            print(f"Error during training of {os.path.basename(script_path)}.")
            print(f"Return code: {process.returncode}")
            # if process.stdout: # Removed as capture_output is False
            #     print("Stdout:\\n", process.stdout)
            # if process.stderr: # Removed as capture_output is False
            #     print("Stderr:\\n", process.stderr)
            print("Check the output above for specific error messages from the script.")
            success = False
        return success

    except FileNotFoundError:
        print(f"Error: The script {script_path} was not found.")
        return False
    except Exception as e:
        print(f"An error occurred while running {script_path}: {e}")
        return False

def main():
    print("--- Master Training Script ---")
    available_scripts = get_available_training_scripts()

    if not available_scripts:
        print("No training scripts found in the 'train' directory. Exiting.")
        return

    print("Available models for training:")
    for i, script_info in enumerate(available_scripts):
        print(f"{i+1}. {script_info['name']}")

    selected_indices = []
    while True:
        try:
            choices_str = input("Enter the numbers of the models to train (comma-separated, e.g., 1,3 or 'all'): ").strip().lower()
            if choices_str == 'all':
                selected_indices = list(range(len(available_scripts)))
                break
            selected_indices = [int(x.strip()) - 1 for x in choices_str.split(',')]
            if all(0 <= idx < len(available_scripts) for idx in selected_indices):
                break
            else:
                print("Invalid selection. Please enter numbers from the list.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas or 'all'.")

    if not selected_indices:
        print("No models selected for training. Exiting.")
        return

    train_file, dev_file, test_file = get_user_data_paths()
    
    print(f"Will use the following datasets for all selected models:")
    print(f"  Train: {train_file}")
    print(f"  Dev:   {dev_file}")
    print(f"  Test:  {test_file}")
    
    confirm = input("Proceed with training? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Training cancelled by user.")
        return

    for idx in selected_indices:
        script_info = available_scripts[idx]
        run_training_script(script_info['path'], train_file, dev_file, test_file)

    print("\\n--- All selected training processes finished. ---")

if __name__ == "__main__":
    main()
