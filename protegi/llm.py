import os

from typing import Dict, List, Union, Literal
from abc import ABC, abstractmethod
from openai import Client
from pydantic import BaseModel

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class LLM(ABC):
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    @abstractmethod
    def chat(self, messages:Union[List[Message], str], *args, **kwargs) -> str:
        pass

class VLLMBackend(LLM):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        api_key = os.getenv("OPENAI_API_KEY", None) or self.cfg.get("api_key", None)
        base_url = os.getenv("OPENAI_API_BASE", None) or self.cfg.get("base_url", None)
        self.model = self.cfg["model"]
        self.client = Client(
            api_key=api_key,
            base_url=base_url,
        )
    
    def chat(self, messages:Union[List[Message], str], *args, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        extra_body = kwargs.get("extra_body", {})
        if "thinking" in self.cfg and not self.cfg["thinking"]:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
        )
        return response.choices[0].message.content.strip()


backend = {
    "vllm": VLLMBackend
}

class AutoLLM:
    @classmethod
    def build(cls, cfg: Dict):
        llm = backend[cfg['backend']](cfg)
        return llm

if __name__ == "__main__":
    cfg = {
        "backend": "vllm",
        "model": "Qwen3-14B",
        "api_key": "sk-proj-1234567890",
        "base_url": "http://localhost:8000/v1",
    }
    llm = AutoLLM.build(cfg)
    print(llm.chat("你好"))