import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re


from core.dataset import Dataset
from datasets import load_dataset
from typing import Dict, List

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

if __name__ == "__main__":
    cfg = {
        "data_path": [
            "/data2/liangguanbao/prompt-optimization/data/gsm8k/main/train-00000-of-00001.parquet",
            "/data2/liangguanbao/prompt-optimization/data/gsm8k/main/test-00000-of-00001.parquet",
        ]
    }
    dataset = GSM8K(cfg)
    from config import supported_llm
    from llm import backend
    llm_cfg = supported_llm["qwen3-14b_vllm"]
    llm = backend[llm_cfg["backend"]](llm_cfg)
    
    # first_example = dataset.split["train"][0]['question']
    # print(first_example)
    # ans = (llm.chat(first_example))
    # print(dataset.model_prediction_postprocess(ans))
    # print(dataset.split["train"][0]['answer'])
    # label = dataset.split["train"][0]['answer']
    # label = dataset.label_postprocess(dataset.split["train"][0]['answer'])
    # print(label)
    # print(dataset.evaluate(ans, label))
