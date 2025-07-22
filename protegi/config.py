
# OPENAI_KEY = "YOUR KEY"

supported_llm = {
    "qwen3-14b_vllm": {
        "backend": "vllm",
        "model": "Qwen3-14B",
        "api_key": "sk-proj-1234567890",
        "base_url": "http://localhost:8000/v1",
        "thinking": True,
    },
    "qwen3-14b_vllm_non-thinking": {
        "backend": "vllm",
        "model": "Qwen3-14B",
        "api_key": "sk-proj-1234567890",
        "base_url": "http://localhost:8000/v1",
        "thinking": False,
    },
}

model = "qwen3-14b_vllm_non-thinking"

do_postprocess = True
