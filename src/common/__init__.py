from common.openai_backend import OpenAIBackend
from common.ollama_backend import OllamaBackend
from common.vllm_backend import VLLMBackend

backend = {
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
    "vllm": VLLMBackend,
}