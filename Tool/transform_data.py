

def transform_to_matchzoo_format(source_data):
    """
    Transforms data from [query, positive_doc, negative_doc] format
    to a list of (query, document, label) tuples.

    Args:
        source_data: A list of lists or tuples, where each inner item
                     is [query, positive_document, negative_document].
                     Example: [
                         ["query1 text", "positive document for query1", "negative document for query1"],
                         ["query2 text", "positive document for query2", "negative document for query2"]
                     ]

    Returns:
        A list of tuples, where each tuple is (query, document, label).
        Label is 1 for positive, 0 for negative.
        Example: [
            ("query1 text", "positive document for query1", 1),
            ("query1 text", "negative document for query1", 0),
            ("query2 text", "positive document for query2", 1),
            ("query2 text", "negative document for query2", 0)
        ]
    """
    transformed_data = []
    for item in source_data:
        query, positive_doc, negative_doc = item
        # Add positive pair
        transformed_data.append((query, positive_doc, 1))
        # Add negative pair
        transformed_data.append((query, negative_doc, 0))
    return transformed_data

# --- Example Usage ---
# Suppose your data is in a list like this:
my_original_data = [
    ["what is the capital of france", "paris is the capital of france", "london is the capital of the uk"],
    ["benefits of python programming", "python is versatile and easy to learn", "java is a compiled language"]
]

# Transform it
matchzoo_ready_data = transform_to_matchzoo_format(my_original_data)

# Print to see the result
for entry in matchzoo_ready_data:
    print(entry)

# You can then use this 'matchzoo_ready_data' to create your pandas DataFrame
# and subsequently the MatchZoo DataPack in your KNRM_training_script.py:
#
# import pandas as pd
# import matchzoo as mz # Assuming mz and ranking_task are defined as in your script
#
# # ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=1)) # Define if not already
#
# if matchzoo_ready_data: # Check if data is not empty
#     # Create DataFrame
#     df = pd.DataFrame(matchzoo_ready_data, columns=['text_left', 'text_right', 'label'])
#
#     # Create DataPack (assuming ranking_task is defined)
#     # data_pack = mz.DataPack(data=df, task=ranking_task)
#     # print(f"Created DataPack with {len(data_pack)} entries.")
# else:
#     print("No data to process.")
