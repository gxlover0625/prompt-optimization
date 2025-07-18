from abc import ABC, abstractmethod
from typing import Dict

class Agent(ABC):
    def __init__(self, cfg:Dict):
        self.cfg = cfg