from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template

import utils
import tasks
import config

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(GPT4Predictor): # Execution Agent
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        dataset_cfg = config.supported_dataset[config.dataset]
        input_key = dataset_cfg["input_key"]
        prompt = Template(prompt).render(**{input_key: ex[input_key]})
        response = utils.chatgpt(
            prompt, max_tokens=1024, n=1, timeout=2, 
            temperature=self.opt['temperature'])[0]
        return response
