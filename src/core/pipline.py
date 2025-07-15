from abc import ABC, abstractmethod
from typing import Dict

class Pipline(ABC):
    def __init__(self, cfg:Dict):
        self.cfg = cfg
    
    @abstractmethod
    def build_pipline(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass