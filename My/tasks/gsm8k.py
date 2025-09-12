import random
import re

from .base import Dataset
from typing import Dict, List
from datasets import load_dataset

# https://github.com/open-compass/opencompass/blob/6101ce61cfb845f5a7269af56e36951f8cae82bd/opencompass/datasets/gsm8k.py#L38C1-L40C51
def label_postprocess(label:str):
    return label.split('#### ')[1].replace(',', '')

# https://github.com/open-compass/opencompass/blob/6101ce61cfb845f5a7269af56e36951f8cae82bd/opencompass/datasets/gsm8k.py#L43C1-L49C23
def model_prediction_postprocess(model_prediction:str)->str:
    text = model_prediction.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]

# https://github.com/open-compass/opencompass/blob/6101ce61cfb845f5a7269af56e36951f8cae82bd/opencompass/datasets/gsm8k.py#L54C5-L60C21
def is_equal(model_prediction:str, label:str) -> float:
    model_prediction = model_prediction_postprocess(model_prediction)
    label = label_postprocess(label)

    try:
        if model_prediction == label or abs(float(model_prediction) - int(label)) < 1e-6:
            return 1.0
    except Exception:
        pass
    return 0.0

class GSM8KBuilder:
    @classmethod
    def build(cls, cfg: Dict):
        train_data = load_dataset("parquet", data_files=cfg["data_path"][0])["train"]
        train_data = [dict(example) for example in train_data]
        test_data = load_dataset("parquet", data_files=cfg["data_path"][1])["train"]
        test_data = [dict(example) for example in test_data]
        
        random.shuffle(train_data)
        random.shuffle(test_data)

        trainset = GSM8K(cfg, train_data[:cfg["train_size"]])
        valset = GSM8K(cfg, train_data[cfg["train_size"]:cfg["train_size"]+cfg["val_size"]])
        if cfg["test_size"] is None:
            cfg["test_size"] = len(test_data)
        testset = GSM8K(cfg, test_data[:cfg["test_size"]])
        return trainset, valset, testset
        
class GSM8K(Dataset):
    def __init__(self, cfg: Dict, dataset: List[Dict], *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.dataset = dataset
    
    def __getitem__(self, index: int):
        question = self.dataset[index]["question"]
        answer = self.dataset[index]["answer"]
        return {"question": question}, answer
    
    def __len__(self):
        return len(self.dataset)
        
if __name__ == "__main__":
    cfg = {
        "data_path": [
            "data/gsm8k/main/train-00000-of-00001.parquet",
            "data/gsm8k/main/test-00000-of-00001.parquet",
        ],
        "train_size": 200,
        "val_size": 300,
        "test_size": None
    }
    trainset, valset, testset = GSM8KBuilder.build(cfg)