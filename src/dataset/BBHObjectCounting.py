import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
from core.dataset import Dataset
from typing import Dict, List

class BBHObjectCounting(Dataset):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.load()
    
    def load(self):
        data_path = self.cfg['data_path']
        self.test_data = self.load_data(data_path)
        self.split = {"test": self.test_data}
    
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

if __name__ == '__main__':
    cfg = {
        "data_path": "/data2/liangguanbao/prompt-optimization/data/bbh/object_counting.json",
        "default_prompt": "Question: {input}\nAnswer:",
        "input_key": "input"
    }
    dataset = BBHObjectCounting(cfg)
    # print(dataset.test_data[0])

    from config import supported_llm
    from llm import AutoLLM
    llm_cfg = supported_llm['qwen3-14b_vllm']
    llm = AutoLLM.build(llm_cfg)
    prompt = dataset.build_prompt(dataset.test_data[0])
    print(prompt)
    print(dataset.test_data[0]['target'])
    print(llm.chat(prompt))
