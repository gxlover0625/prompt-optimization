import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import LLM, Message
from typing import List, Dict, Union
from ollama import chat

class OllamaBackend(LLM):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.model = self.cfg["model"]
    
    def chat(self, messages:Union[List[Message], str], *args, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        if "thinking" in self.cfg and not self.cfg["thinking"]:
            think = False
        else:
            think = True

        response = chat(
            model=self.model,
            messages=messages,
            think=think,
        )
        return response.message.content
    

if __name__ == "__main__":
    cfg = {
        "model": "qwen2:7b",
    }
    llm = OllamaBackend(cfg)
    response = llm.chat("Hello, how are you?")
    print(response)