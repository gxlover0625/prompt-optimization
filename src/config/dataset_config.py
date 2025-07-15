supported_dataset = {
    "liar": {
        "dataset_name": "Liar",
        "data_path": [
            "data/liar/train.jsonl",
            "data/liar/test.jsonl",
        ],
        "label_key": "label",
        "input_key": "text",
        "default_prompt": "# Task\nDetermine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {text}\nLabel:"
    },
    "gsm8k": {
        "dataset_name": "GSM8K",
        "data_path": [
            "data/gsm8k/main/train-00000-of-00001.parquet",
            "data/gsm8k/main/test-00000-of-00001.parquet",
        ],
        "label_key": "answer",
        "input_key": "question",
        "default_prompt": "Question: {question}\nAnswer:"
    }
}