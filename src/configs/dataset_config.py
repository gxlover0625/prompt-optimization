supported_dataset = {
    "liar": {
        "data_path": [
            "data/liar/train.jsonl",
            "data/liar/test.jsonl",
        ],
        "label_key": "label",
        "input_key": "text",
        "default_prompt": "# Task\nDetermine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {text}\nLabel:"
    }
}