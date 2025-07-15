import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from config import supported_llm, supported_dataset
from dataset import AutoDataset
from core.agent import Agent
from core.pipline import Pipline
from llm import backend

from typing import Dict, List

class ExecutionAgent(Agent):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.llm = backend[cfg["backend"]](cfg)
    
    def execute(self, example:Dict):
        pass

    def run(self, examples:List[Dict]):
        pass

class DirectPipline(Pipline):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.build_pipline()
    
    def build_pipline(self):
        self.execution_agent = ExecutionAgent(self.cfg)
        self.dataset = AutoDataset.build_dataset(self.cfg)
    
    def run(self):
        results = []
        for idx, example in enumerate(self.dataset.split["test"], 1):
            messages = self.dataset.build_prompt(example)
            response = self.execution_agent.llm.chat(messages)
            results.append({
                "idx": f"{self.cfg['model']}_{self.cfg['dataset_name']}_{idx}",
                "prompt": messages,
                "model_prediction": response,
                "label": example[self.cfg["label_key"]],
            })
            print(results)
            break
        pass

if __name__ == "__main__":
    llm_cfg = supported_llm["qwen3-14b_vllm"]
    # agent = ExecutionAgent(llm_cfg)
    # print(agent.llm.chat("你好"))

    dataset_cfg = supported_dataset["liar"]
    # dataset = Liar(dataset_cfg)
    # print(len(dataset.split["train"]),len(dataset.split["test"]))
    # print(dataset.split["train"][0])

    ## 将llm_cfg和dataset_cfg合并成pipline_cfg
    pipline_cfg = {**llm_cfg, **dataset_cfg}
    # print(pipline_cfg)
    pipline = DirectPipline(pipline_cfg)
    # print(pipline.execution_agent.llm.chat("你好"))

    pipline.run()
