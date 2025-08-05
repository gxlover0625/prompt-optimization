from abc import ABC, abstractmethod
from typing import Dict, List
from datasets import load_dataset
import re
import os
import json
import random

class Dataset(ABC):
    def __init__(self, cfg:Dict):
        self.cfg = cfg
    
    @abstractmethod
    def build_prompt(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

class GSM8K(Dataset):
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

    def load_data(self, data_path:str)->List[Dict]:
        dataset = load_dataset("parquet", data_files=data_path)
        return dataset["train"]

    def build_prompt(self, example:Dict):
        default_prompt = self.cfg["default_prompt"]
        input_key = self.cfg["input_key"]
        prompt = default_prompt.format(**{input_key: example[input_key]})
        return prompt

    # copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
    def label_postprocess(self, label:str):
        return label.split('#### ')[1].replace(',', '')
    
    # copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
    def model_prediction_postprocess(self, model_prediction:str)->str:
        text = model_prediction.split('Question:')[0]
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
        if not numbers:
            return 'NULL'
        return numbers[-1]
    
    # copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
    def is_equal(self, model_prediction:str, label:str) -> bool:
        try:
            if model_prediction == label or abs(float(model_prediction) - int(label)) < 1e-6:
                return True
        except Exception:
            pass
        return False
    
    def evaluate(self, model_prediction:str, label:str) -> bool:
        model_prediction = self.model_prediction_postprocess(model_prediction)
        label = self.label_postprocess(label)
        return self.is_equal(model_prediction, label)

class BBHObjectCounting(Dataset):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.load()
    
    def load(self):
        data_path = self.cfg['data_path']
        self.data = self.load_data(data_path)
        if "train_ratio" in self.cfg:
            self.split_data(self.cfg['train_ratio'])
        else:
            self.test_data = self.data
            self.train_data = []

    def split_data(self, train_ratio:float=0.7):
        random.shuffle(self.data)
        train_size = int(len(self.data) * train_ratio)
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:] 
    
    def load_data(self, data_path:str)->List[Dict]:
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data['examples']
    
    def build_prompt(self, example:Dict):
        default_prompt = self.cfg["default_prompt"]
        input_key = self.cfg["input_key"]
        prompt = default_prompt.format(**{input_key: example[input_key]})
        return prompt
    
    def label_postprocess(self, label:str):
        return label

    # copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
    def model_prediction_postprocess(self, model_prediction:str)->str:
        text = model_prediction.split('Question:')[0]
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
        if not numbers:
            return 'NULL'
        return numbers[-1]

    # copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
    def is_equal(self, model_prediction:str, label:str):
        try:
            if model_prediction == label or abs(float(model_prediction) - int(label)) < 1e-6:
                return True
        except Exception:
            pass
        return False
    
    def evaluate(self, model_prediction:str, label:str):
        model_prediction = self.model_prediction_postprocess(model_prediction)
        label = self.label_postprocess(label)
        return self.is_equal(model_prediction, label)

dataset_mapping = {
    "GSM8K": GSM8K,
    "BBHObjectCounting": BBHObjectCounting,
}

class AutoDataset:
    @classmethod
    def build(cls, cfg:Dict):
        dataset_name = cfg["dataset_name"]
        dataset = dataset_mapping[dataset_name](cfg)
        return dataset

if __name__ == "__main__":
    cfg = {
        "dataset_name": "GSM8K",
        "data_path": [
            "data/gsm8k/main/train-00000-of-00001.parquet",
            "data/gsm8k/main/test-00000-of-00001.parquet",
        ],
        "label_key": "answer",
        "input_key": "question",
        "default_prompt": "Question: {question}\nAnswer:"
    }
    # dataset = GSM8K(cfg)
    # dataset = AutoDataset.build(cfg)
    # print(dataset.train_data[0])
    # print(len(dataset.train_data), len(dataset.test_data))