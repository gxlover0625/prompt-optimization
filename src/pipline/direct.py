import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import supported_llm, supported_dataset
from dataset import Liar
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
    
    def run(self):
        pass

if __name__ == "__main__":
    llm_cfg = supported_llm["qwen3-14b_vllm"]
    agent = ExecutionAgent(llm_cfg)
    print(agent.llm.chat("你好"))

    dataset_cfg = supported_dataset["liar"]
    dataset = Liar(dataset_cfg)
    print(len(dataset.split["train"]),len(dataset.split["test"]))
    print(dataset.split["train"][0])
