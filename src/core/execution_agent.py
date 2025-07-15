import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
from core.agent import Agent
from common import backend

class ExecutionAgent(Agent):
    def __init__(self, cfg: Dict):
        super().__init__(cfg=cfg)
        self.llm = backend[cfg['backend']](cfg)
    
    def run(self, *args, **kwargs):
        return self._execute(*args, **kwargs)
    
    def _execute(self, examples:List[Dict]):
        pass


if __name__ == "__main__":
    cfg = {
        "backend": "ollama",
        "model": "qwen2:7b",
        "api_key": "sk-proj-1234567890",
        "base_url": "http://localhost:11434/v1",
    }
    agent = ExecutionAgent(cfg)
    print(agent.llm.chat("Hello, how are you?"))