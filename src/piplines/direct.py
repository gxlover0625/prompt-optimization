import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import supported_llm
from typing import Dict
from common import backend

class DirectPipeline:
    def __init__(self, cfg:Dict):
        self.cfg = cfg

if __name__ == "__main__":
    llm_cfg = supported_llm["qwen3-14b_vllm"]
    print(llm_cfg)
    llm = backend[llm_cfg["backend"]](llm_cfg)
    print(llm.chat("你好"))