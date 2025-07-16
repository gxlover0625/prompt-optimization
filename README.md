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

## 📊 Data Preparation
Please refer to the [docs/prepare_dataset.md](docs/prepare_dataset.md).

## 📖 Model Preparation
Please refer to the [docs/prepare_model.md](docs/prepare_model.md).

## 🤝 Acknowledgements
We were inspired by the excellent open-source project [OpenCompass](https://github.com/open-compass/opencompass), which helped simplify our development. Additionally, we would like to thank the following open-source projects for their code contributions.
- [OpenCompass](https://github.com/open-compass/opencompass) ![Star](https://img.shields.io/github/stars/open-compass/opencompass.svg?style=social&label=Star), is an LLM evaluation platform, supporting a wide range of models (Llama3, Mistral, InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets.
- [LMOps](https://github.com/microsoft/LMOps) ![Star](https://img.shields.io/github/stars/microsoft/LMOps.svg?style=social&label=Star), general technology for enabling AI capabilities w/ LLMs and MLLMs