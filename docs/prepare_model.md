# Model Preparation

## Supported Backends
- [vllm](https://docs.vllm.ai/en/latest/getting_started/quickstart.html), A high-throughput and memory-efficient inference and serving engine for LLMs
- [ollama](https://ollama.com/), get up and running with Llama 3.3, DeepSeek-R1, Phi-4, Gemma 3, Mistral Small 3.1 and other large language models.
- [OpenAI-Compatible](https://github.com/openai/openai-python), the official Python library for the OpenAI API

## vllm
### Installation
Please refer to the [vllm installation](https://docs.vllm.ai/en/latest/getting_started/installation/index.html).
### Usage
There is a example of using vllm to run [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B).  
First, you need to download the model from huggingface. If you have already downloaded the model, you can skip this step.
```bash
# download the model
# If you cannot connect to huggingface, we recommend to use mirror website https://hf-mirror.com/
export HF_ENDPOINT=https://hf-mirror.com
```
