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

## GSM8K
We used the dataset from https://huggingface.co/datasets/openai/gsm8k. Below are the commands to run on a Linux system: