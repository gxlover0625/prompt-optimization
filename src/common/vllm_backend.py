import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import LLM, Message
from typing import List, Dict, Union
from openai import Client

## copy from openai_agent.py
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
        
        extra_body = {}
        if not self.cfg["thinking"]:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
        )
        return response.choices[0].message.content
    
if __name__ == "__main__":
    cfg = {
        "model": "qwen2:7b",
        "api_key": "sk-proj-1234567890",
        "base_url": "http://localhost:11434/v1",
    }
    llm = VLLMBackend(cfg)
    response = llm.chat("你好")
    print(response)