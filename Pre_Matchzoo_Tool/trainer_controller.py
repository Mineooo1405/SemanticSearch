import os
import sys

try:
    import master_trainer
except ImportError:
    print("Error: master_trainer.py not found. Make sure it's in the same directory as controller.py or in PYTHONPATH.")
    sys.exit(1)
DATASET_CONFIGURATIONS = [
    {
        "name": "msmarco_semantic-grouping",
        "train_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_1/msmarco_semantic-grouping_train_train_mz.tsv",
        "dev_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_1/msmarco_semantic-grouping_train_dev_mz.tsv",
        "test_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_1/msmarco_semantic-grouping_test.tsv",
        "models_to_train": "all" 
    },
    {
        "name": "msmarco_semantic-grouping_oie",
        "train_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping_oie/train_1/msmarco_semantic-grouping_oie_train_train_mz.tsv",
        "dev_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping_oie/train_1/msmarco_semantic-grouping_oie_train_dev_mz.tsv",
        "test_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping_oie/train_1/msmarco_semantic-grouping_oie_test.tsv",
        #"models_to_train": ["MatchLSTM"] 
        #"models_to_train": ["KNRM", "ConvKNRM"] 
        "models_to_train": "all"
    },
    {
        "name": "msmarco_semantic-splitter",
        "train_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-splitter/train_1/msmarco_semantic-splitter_train_train_mz.tsv",
        "dev_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-splitter/train_1/msmarco_semantic-splitter_train_dev_mz.tsv",
        "test_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-splitter/train_1/msmarco_semantic-splitter_test.tsv",
        "models_to_train": "all"
    },
    {
        "name": "msmarco_semantic-splitter_oie",
        "train_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-splitter_oie/train_1/msmarco_semantic-splitter_oie_train_train_mz.tsv",
        "dev_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-splitter_oie/train_1/msmarco_semantic-splitter_oie_train_dev_mz.tsv",
        "test_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-splitter_oie/train_1/msmarco_semantic-splitter_oie_test.tsv",
        "models_to_train": "all"
    },
    {
        "name": "msmarco_text-splitter",
        "train_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_text-splitter/train_1/msmarco_text-splitter_train_train_mz.tsv",
        "dev_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_text-splitter/train_1/msmarco_text-splitter_train_dev_mz.tsv",
        "test_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_text-splitter/train_1/msmarco_text-splitter_test.tsv",
        "models_to_train": "all"
    },
    {
        "name": "msmarco_text-splitter_oie",
        "train_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_text-splitter_oie/train_1/msmarco_text-splitter_oie_train_train_mz.tsv",
        "dev_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_text-splitter_oie/train_1/msmarco_text-splitter_oie_train_dev_mz.tsv",
        "test_path": "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_text-splitter_oie/train_1/msmarco_text-splitter_oie_test.tsv",
        "models_to_train": "all"
    }
]

def main_controller():
    print("--- Training Controller Script ---")

    all_available_scripts = master_trainer.get_available_training_scripts()
    if not all_available_scripts:
        print(f"No training scripts found in '{master_trainer.TRAIN_SCRIPTS_DIR}'. Exiting controller.")
        return

    print("\nAvailable models found by master_trainer:")
    for script_info in all_available_scripts:
        print(f"- Model: {script_info['name']}") # Path: {script_info['path']}
    print("-" * 30)

    for i, config in enumerate(DATASET_CONFIGURATIONS):
        print(f"\n--- Processing Dataset Configuration {i+1}/{len(DATASET_CONFIGURATIONS)}: {config['name']} ---")
        print(f"  Train dataset: {config['train_path']}")
        print(f"  Dev dataset:   {config['dev_path']}")
        print(f"  Test dataset:  {config['test_path']}")

        paths_valid = True
        for path_key in ["train_path", "dev_path", "test_path"]:
            current_path = config[path_key]
            if not os.path.isabs(current_path):
                print(f"Warning: Path for {path_key} ('{current_path}') is not absolute. master_trainer expects absolute paths.")

            if not os.path.isfile(current_path):
                print(f"Error: File not found for {path_key}: {current_path}")
                paths_valid = False
        
        if not paths_valid:
            print(f"Skipping dataset configuration '{config['name']}' due to missing or invalid file paths.")
            continue

        models_to_run_for_this_config = []
        if isinstance(config["models_to_train"], str) and config["models_to_train"].lower() == "all":
            models_to_run_for_this_config = all_available_scripts
        elif isinstance(config["models_to_train"], list):
            for model_name_to_train in config["models_to_train"]:
                found_model = False
                for script_info in all_available_scripts:
                    if script_info["name"] == model_name_to_train:
                        models_to_run_for_this_config.append(script_info)
                        found_model = True
                        break
                if not found_model:
                    print(f"Warning: Model '{model_name_to_train}' specified in config for dataset '{config['name']}' but not found in available scripts. It will be skipped for this dataset.")
        else:
            print(f"Warning: Invalid 'models_to_train' format for dataset '{config['name']}'. Expected 'all' or a list of model names. Skipping this dataset configuration.")
            continue
            
        if not models_to_run_for_this_config:
            print(f"No models selected or found for dataset configuration: '{config['name']}'. Skipping.")
            continue

        print(f"Models to train for this dataset configuration: {[m['name'] for m in models_to_run_for_this_config]}")

        for script_info in models_to_run_for_this_config:
            print(f"\nStarting training for model: '{script_info['name']}' with dataset: '{config['name']}'")
            success = master_trainer.run_training_script(
                script_path=script_info['path'],
                train_file=config['train_path'],
                dev_file=config['dev_path'],
                test_file=config['test_path']
            )
            if success:
                print(f"Successfully completed training for model '{script_info['name']}' with dataset '{config['name']}'.")
            else:
                print(f"Training failed or encountered an error for model '{script_info['name']}' with dataset '{config['name']}'.")
        
        print(f"--- Finished processing Dataset Configuration: {config['name']} ---")

    print("\n--- All dataset configurations processed by controller. ---")

if __name__ == "__main__":
    main_controller()