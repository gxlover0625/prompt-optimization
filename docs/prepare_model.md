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
- First, you need to download the model from huggingface. If you have already downloaded the model, you can skip this step.
```bash
# If you cannot connect to huggingface, we recommend to use mirror website https://hf-mirror.com/
export HF_ENDPOINT=https://hf-mirror.com

# install the requirements
pip install -U huggingface_hub

cd prompt-optimization
mkdir -p models/Qwen/Qwen3-14B

huggingface-cli download --resume-download Qwen/Qwen3-14B --local-dir ./models/Qwen/Qwen3-14B
```
- Second, you can use the following command to run the model. If you are familar with vllm, you can adjust the parameters to fit your needs.
```bash
vllm serve models/Qwen/Qwen3-14B \
    --served-model-name Qwen3-14B \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --reasoning-parser qwen3 \
    --tensor-parallel-size 2
```
- Third, you need to modify the [llm_config.py](../src/config/llm_config.py) to add the model you want to use.
```python
{
    "qwen3-14b_vllm": {
        "backend": "vllm", # the backend you want to use, there are there choices: vllm, ollama, openai
        "model": "Qwen3-14B", # the model you want to use, according to the served-model-name parameter in the vllm command
        "api_key": "sk-proj-1234567890", # it can be any string
        "base_url": "http://localhost:8000/v1", # according to the port parameter in the vllm command
        "thinking": True, # whether to use thinking ability of the model, not all models support this
    }
}
```