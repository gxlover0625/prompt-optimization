import os

from pydantic import BaseModel
from typing import Literal, Union, List
from openai import OpenAI

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class OpenAIClient:
    def __init__(self, model: str, api_key: str=None, base_url: str=None):
        self.model = model
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url
        )
    
    def chat(self, messages:Union[List[Message], str], *args, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        extra_body = kwargs.get("extra_body", None)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1.0,
            extra_body=extra_body,
        )
        return response.choices[0].message.content