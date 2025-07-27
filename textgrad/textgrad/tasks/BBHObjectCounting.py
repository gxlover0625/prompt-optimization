import json
import random
import re
import textgrad as tg
from typing import Dict, List

def label_postprocess(label:str):
    return label

# copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
def model_prediction_postprocess(model_prediction:str)->str:
    text = model_prediction.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]

# copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
def is_equal(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    model_prediction_str = model_prediction_postprocess(str(prediction.value))
    label_str = label_postprocess(str(ground_truth_answer.value))
    try:
        if model_prediction_str == label_str or abs(float(model_prediction_str) - int(label_str)) < 1e-6:
            return 1
    except Exception:
        pass
    return 0

class BBHObjectCounting:
    def __init__(self, cfg:Dict, split="train"):
        self.cfg = cfg
        self.split = split
        self._task_description = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
        self.load()
    
    def get_task_description(self):
        return self._task_description
    
    def __getitem__(self, index):
        if self.split == "train":
            cur_example = self.train_data[index]
            return cur_example[self.cfg['input_key']], cur_example[self.cfg['label_key']]
        else:
            cur_example = self.test_data[index]
            return cur_example[self.cfg['input_key']], cur_example[self.cfg['label_key']]
    
    def __len__(self):
        return len(self.train_data) if self.split == "train" else len(self.test_data)
    
    def load(self):
        data_path = self.cfg['data_path']
        self.data = self.load_data(data_path)
        if "train_ratio" in self.cfg:
            self.split_data(self.cfg['train_ratio'])
        else:
            self.test_data = self.data
            self.train_data = []
        
    def load_data(self, data_path:str)->List[Dict]:
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data['examples']
    
    def split_data(self, train_ratio:float=0.7):
        random.shuffle(self.data)
        train_size = int(len(self.data) * train_ratio)
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:] 
        for idx, sample in enumerate(self.train_data):
            self.train_data[idx]["id"] = f"train-{idx}"
        for idx, sample in enumerate(self.test_data):
            self.test_data[idx]["id"] = f"test-{idx}"


if __name__ == "__main__":
    dataset_cfg = {
        "dataset_name": "BBHObjectCounting",
        "data_path": "data/bbh/object_counting.json",
        "label_key": "target",
        "input_key": "input",
        "default_prompt": "Question: {input}\nAnswer:",
        "train_ratio": 0.7
    }
    dataset = BBHObjectCounting(dataset_cfg)
