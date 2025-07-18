import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.pipline import Pipline
from ref.protegi.main import get_args
from typing import Dict

class ProTeGiPipline(Pipline):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
    
    def build_pipline(self):
        pass

    def run(self):
        pass

if __name__ == "__main__":
    import argparse
    args = get_args()
    print(args)
    pass