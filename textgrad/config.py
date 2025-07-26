
# OPENAI_KEY = "YOUR KEY"

supported_llm = {
    "qwen3-14b_vllm": {
        "backend": "vllm",
        "model": "Qwen3-14B",
        "api_key": "sk-proj-1234567890",
        "base_url": "http://localhost:8000/v1",
        "thinking": True,
    },
    "qwen3-14b_vllm_non-thinking": {
        "backend": "vllm",
        "model": "Qwen3-14B",
        "api_key": "sk-proj-1234567890",
        "base_url": "http://localhost:8000/v1",
        "thinking": False,
    },
}

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
    },
    "bbh_object_counting": {
        "dataset_name": "BBHObjectCounting",
        "data_path": "data/bbh/object_counting.json",
        "label_key": "target",
        "input_key": "input",
        "default_prompt": "Question: {input}\nAnswer:",
        "train_ratio": 0.7
    }
}

execution_agent = "qwen3-14b_vllm_non-thinking"
evaluation_agent = "default"
optimization_agent = "qwen3-14b_vllm_non-thinking"