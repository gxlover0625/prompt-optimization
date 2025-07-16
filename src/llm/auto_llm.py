from typing import Dict

from llm.openai_backend import OpenAIBackend
from llm.ollama_backend import OllamaBackend
from llm.vllm_backend import VLLMBackend

backend = {
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
    "vllm": VLLMBackend,
}

class AutoLLM:
    @classmethod
    def build(cls, cfg: Dict):
        llm = backend[cfg['backend']](cfg)
        return llm
