supported_dataset = {
    "liar": {
        "dataset_name": "Liar",
        "data_path": [
            "data/liar/train.jsonl",
            "data/liar/test.jsonl",
        ],
        "label_key": "label",
        "input_key": "text",
        "default_prompt": "# Task\nDetermine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {text}\nLabel:",
        "protegi_prompt": "# Task\nDetermine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\n# Output format\nAnswer Yes or No as labels. Ensure the response concludes with the answer in the format: <answer>answer</answer>.\n\n# Prediction\nText: {{ text }}\nLabel:\n"   
    },
    "gsm8k": {
        "dataset_name": "GSM8K",
        "data_path": [
            "data/gsm8k/main/train-00000-of-00001.parquet",
            "data/gsm8k/main/test-00000-of-00001.parquet",
        ],
        "label_key": "answer",
        "input_key": "question",
        "default_prompt": "Question: {question}\nAnswer:",
        "protegi_prompt": "# Task\nSolve the following math problem.\n\n# Output format\nEnsure the response concludes with the answer in the format: <answer>answer</answer>.\n\n# Prediction\nQuestion: {{ question }}\nAnswer:"
    },
    "bbh_object_counting": {
        "dataset_name": "BBHObjectCounting",
        "data_path": "data/bbh/object_counting.json",
        "label_key": "target",
        "input_key": "input",
        "default_prompt": "Question: {input}\nAnswer:",
        "protegi_prompt": "# Task\nSolve the following problem.\n\n# Output format\nEnsure the response concludes with the answer in the format: <answer>answer</answer>.\n\n# Prediction\nQuestion: {{ question }}\nAnswer:"
    }
}