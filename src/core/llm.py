from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Literal, Dict, Union

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class LLM(ABC):
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    @abstractmethod
    def chat(self, messages:Union[List[Message], str], *args, **kwargs) -> str:
        pass