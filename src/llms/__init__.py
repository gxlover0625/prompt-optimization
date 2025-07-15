from llms.openai_backend import OpenAIBackend
from llms.ollama_backend import OllamaBackend
from llms.vllm_backend import VLLMBackend

backend = {
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
    "vllm": VLLMBackend,
}