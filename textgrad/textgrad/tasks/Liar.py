import json
import textgrad as tg
from typing import Dict, List

def label_postprocess(label:str):
    return label

def model_prediction_postprocess(model_prediction:str)->str:
    if model_prediction.strip().upper().startswith('YES'):
        return "Yes"
    else:
        return "No"

def is_equal(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    model_prediction_str = model_prediction_postprocess(str(prediction.value))
    label_str = label_postprocess(str(ground_truth_answer.value))
    try:
        if model_prediction_str == label_str:
            return 1
        else:
            return 0
    except Exception:
        return 0

class Liar:
    def __init__(self, cfg:Dict, split="train"):
        self.cfg = cfg
        self.split = split
        self._task_description = "# Task\nDetermine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.\n\n# Output format\nAnswer Yes or No as labels"
        self.load()
    
    def get_task_description(self):
        return self._task_description
    
    def load(self):
        data_path = self.cfg['data_path']
        self.data = self.load_data(data_path)
        self.train_data = self.data['train']
        self.test_data = self.data['test']

    def load_data(self, data_path: List[str]):
        train_data = []
        with open(data_path[0], "r") as f:
            for line in f:
                train_data.append(json.loads(line))
        test_data = []
        with open(data_path[1], "r") as f:
            for line in f:
                test_data.append(json.loads(line))
        return {"train": train_data, "test": test_data}

    def __len__(self):
        return len(self.train_data) if self.split == "train" else len(self.test_data)
    
    def __getitem__(self, index):
        if self.split == "train":
            cur_example = self.train_data[index]
            label = cur_example[self.cfg['label_key']]
            if str(label) == '0':
                label = 'No'
            else:
                label = 'Yes'
            return cur_example[self.cfg['input_key']], label
        else:
            cur_example = self.test_data[index]
            label = cur_example[self.cfg['label_key']]  
            if str(label) == '0':
                label = 'No'
            else:
                label = 'Yes'
            return cur_example[self.cfg['input_key']], label

if __name__ == "__main__":
    cfg = {
        "data_path": ["data/liar/train.jsonl", "data/liar/test.jsonl"],
        "input_key": "text",
        "label_key": "label"
    }
    liar = Liar(cfg, split="train")
    print(len(liar))
    print(liar[2])