# Dataset Preparation

## Supported Datasets
- Liar
- GSM8K

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