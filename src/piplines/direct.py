import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import supported_llm
from typing import Dict

class DirectPipeline:
    def __init__(self, cfg:Dict):
        self.cfg = cfg

if __name__ == "__main__":
    print(supported_llm)