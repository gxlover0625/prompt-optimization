import re
import textgrad as tg

from typing import Dict, List
from datasets import load_dataset


# copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
def label_postprocess(label:str):
    return label.split('#### ')[1].replace(',', '')

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

class GSM8K:
    def __init__(self, cfg:Dict, split="train") -> None:
        self.cfg = cfg
        self.split = split
        self._task_description = "You will answer a mathemetical reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
        self.load()
    
    def get_task_description(self):
        return self._task_description
    
    def load(self):
        data_path = self.cfg['data_path']
        self.data = self.load_data(data_path)
        self.train_data = self.data['train']
        self.test_data = self.data['test']
    
    def load_data(self, data_path: List[str]):
        train_data = load_dataset("parquet", data_files=data_path[0])['train']
        test_data = load_dataset("parquet", data_files=data_path[1])['train']
        return {"train": train_data, "test": test_data}
    
    def __len__(self):
        return len(self.train_data) if self.split == "train" else len(self.test_data)
    
    def __getitem__(self, index):
        if self.split == "train":
            cur_example = self.train_data[index]
            return cur_example[self.cfg['input_key']], cur_example[self.cfg['label_key']]
        else:
            cur_example = self.test_data[index]
            return cur_example[self.cfg['input_key']], cur_example[self.cfg['label_key']]
        
if __name__ == "__main__":
    cfg = {
        "data_path": ["data/gsm8k/main/train-00000-of-00001.parquet", "data/gsm8k/main/test-00000-of-00001.parquet"],
        "input_key": "question",
        "label_key": "answer",
        "default_prompt": "Question: {question}\nAnswer:"
    }
    gsm8k = GSM8K(cfg, split="train")
    print(gsm8k[0])