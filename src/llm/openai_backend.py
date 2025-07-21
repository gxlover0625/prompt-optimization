import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import Client
from core.llm import LLM, Message
from typing import List, Dict, Union

class OpenAIBackend(LLM):
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
        # TODO: add thinking

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        extra_body = kwargs.get("extra_body", {})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
        )
        return response.choices[0].message.content
        

if __name__ == "__main__":
    cfg = {
        "model": "Qwen3-14B",
        "api_key": "sk-proj-1234567890",
        "base_url": "http://localhost:8000/v1",
    }
    llm = OpenAIBackend(cfg)
    response = llm.chat("Hello, how are you?", extra_body={"temperature": 0.23, "top_p":0.88, "stream": False})
    print(response)
