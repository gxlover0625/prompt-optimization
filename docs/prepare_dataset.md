# Dataset Preparation

## Supported Datasets
> [!NOTE]
> You don't need to download all the datasets at once. Please follow the instructions below to download the dataset you need.
- Liar
- GSM8K
- BBHObjectCounting

## Liar
We used the dataset from https://github.com/microsoft/LMOps/tree/main/prompt_optimization/data/liar. Below are the commands to run on a Linux system:
```bash
cd prompt-optimization
mkdir -p data/liar
cd data/liar
wget https://raw.githubusercontent.com/microsoft/LMOps/main/prompt_optimization/data/liar/train.jsonl
wget https://raw.githubusercontent.com/microsoft/LMOps/main/prompt_optimization/data/liar/test.jsonl
```
After download the dataset, you will have the following directory structure:
```
prompt-optimization/
├── data/
│   ├── liar/
│   │   ├── train.jsonl
│   │   └── test.jsonl
```
> [!IMPORTANT]
> If you don't download the dataset in the `data` directory, you should modify the `data_path` parameter in the `src/config/dataset_config.py`. 
```python
"liar": {
    "dataset_name": "Liar",
    "data_path": [
        "data/liar/train.jsonl", # here, you need to modify the absolute path of the dataset, train.jsonl first
        "data/liar/test.jsonl", # here, you need to modify the absolute path of the dataset, test.jsonl second
    ],
    "label_key": "label",
    "input_key": "text",
    "default_prompt": "# Task\nDetermine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\n# Output format\nAnswer Yes or No as labels\n\n# Prediction\nText: {text}\nLabel:"
}
```

## GSM8K
We used the dataset from https://huggingface.co/datasets/openai/gsm8k. Below are the commands to run on a Linux system:
```bash
cd prompt-optimization
mkdir -p data/gsm8k
cd data/gsm8k

# If you cannot connect to huggingface, we recommend to use mirror website https://hf-mirror.com/
export HF_ENDPOINT=https://hf-mirror.com

# install the requirements
pip install -U huggingface_hub

# download the dataset
huggingface-cli download --repo-type dataset --resume-download openai/gsm8k --local-dir ./
```
After download the dataset, you will have the following directory structure:
```
prompt-optimization/
├── data/
│   ├── gsm8k/
│   │   ├── main/
│   │   │   ├── test--00000-of-00001.parquet
│   │   │   └── train--00000-of-00001.parquet
```
> [!IMPORTANT]
> If you don't download the dataset in the `data` directory, you should modify the `data_path` parameter in the `src/config/dataset_config.py`. 
```python
"gsm8k": {
    "dataset_name": "GSM8K",
    "data_path": [
        "data/gsm8k/main/train-00000-of-00001.parquet", # here, you need to modify the absolute path of the dataset, train first
        "data/gsm8k/main/test-00000-of-00001.parquet", # here, you need to modify the absolute path of the dataset, test second
    ],
    "label_key": "answer",
    "input_key": "question",
    "default_prompt": "Question: {question}\nAnswer:"
}
```

## BBHObjectCounting
We used the dataset from https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh. Below are the commands to run on a Linux system:
```bash
cd prompt-optimization
mkdir -p data/bbh
cd data/bbh
wget https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/object_counting.json
```
After download the dataset, you will have the following directory structure:
```
prompt-optimization/
├── data/
│   ├── bbh/
│   │   └── object_counting.json
```
> [!IMPORTANT]
> If you don't download the dataset in the `data` directory, you should modify the `data_path` parameter in the `src/config/dataset_config.py`. 
```python
"bbh_object_counting": {
    "dataset_name": "BBHObjectCounting",
    "data_path": "data/bbh/object_counting.json", # here, you need to modify the absolute path of the dataset, object_counting.json first
    "label_key": "target",
    "input_key": "input",
    "default_prompt": "Question: {input}\nAnswer:"
}
```