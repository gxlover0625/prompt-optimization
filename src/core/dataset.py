from abc import ABC, abstractmethod
from typing import Dict

class Dataset(ABC):
    def __init__(self, cfg:Dict):
        self.cfg = cfg
        self.data = None
    
    @abstractmethod
    def build_prompt(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass