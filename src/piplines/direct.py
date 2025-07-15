import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import supported_llm
from typing import Dict
from core.agent import Agent
from common import backend

class ExecutionAgent(Agent):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.llm = backend[cfg["backend"]](cfg)
    
    def run(self, *args, **kwargs):
        pass

if __name__ == "__main__":
    llm_cfg = supported_llm["qwen3-14b_vllm"]
    agent = ExecutionAgent(llm_cfg)
    print(agent.llm.chat("你好"))