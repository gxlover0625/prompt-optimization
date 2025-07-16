import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from core.dataset import Dataset
from typing import Dict, List

class Liar(Dataset):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.load()
    
    def load(self):
        data_path = self.cfg["data_path"]
        if isinstance(data_path, str):
            self.train_data = self.load_data(data_path)
            self.split = {"train": self.train_data, "test": None}
        elif isinstance(data_path, List):
            self.train_data = self.load_data(data_path[0])
            self.test_data = self.load_data(data_path[1])
            self.split = {"train": self.train_data, "test": self.test_data}
    
    def load_data(self, data_path:str)-> List[Dict]:
        data = []
        with open(data_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def build_prompt(self, example:Dict):
        default_prompt = self.cfg["default_prompt"]
        input_key = self.cfg["input_key"]
        prompt = default_prompt.format(**{input_key: example[input_key]})
        return prompt
    
    # copy from https://github.com/microsoft/LMOps/blob/main/prompt_optimization/predictors.py
    def evaluate(self, model_prediction:str, label:int):
        extracted_prediction = 1 if model_prediction.strip().upper().startswith('YES') else 0
        return extracted_prediction == label

if __name__ == "__main__":
    cfg = {
        "data_path": [
            "data/liar/train.jsonl",
            "data/liar/test.jsonl",
        ]
    }
    dataset = Liar(cfg)
    train_data = dataset.split["train"]
    test_data = dataset.split["test"]
    print(len(train_data), len(test_data))
    # print(dataset.data)