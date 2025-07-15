# prompt-optimization

## Update
0714, agent base & chat function
```python
from common import backend

cfg = {
    "backend": "ollama",
    "model": "qwen2:7b",
    "api_key": "sk-proj-1234567890",
    "base_url": "http://localhost:11434/v1",
}

llm = backend[cfg['backend']](cfg)
print(llm.chat("你好"))
```
- core是抽象类，主要有LLM基类和它的chat接口、Agent基类和它的run接口、ExecutionAgent基类和它的execute接口
- common是具体实现，包括LLM基类的三个具体实现，ollama、openai、vllm
- dataset处理数据
- pipline对应完整方法