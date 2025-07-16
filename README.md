# prompt-optimization



## 🚀 What's New
## todo
- [ ] add the `chain-of-thought` pipline
## 🔥🔥 0716-Night
- We add the `AutoPipline` class to automatically instantiate the pipline according to the `pipline` parameter in the `src/main.py`. For more details, please refer to the `src/config/pipline_config.py`.
- We add the `examples` directory to provide the examples of how to use the project.
- We design more parameters in `src/main.py` to support more diverse piplines.

## 🔥🔥🔥 0716-Afternoon
> [!IMPORTANT]
> Only vllm and ollama backend support the reasoning model now.
- We support the reasoning model like Qwen3 series etc. Just set the `thinking` parameter in the `src/config/llm_config.py` to `True`.
- We support the `direct pipline` now. Model will directly output the result without `prompt optimization`, which serves as a baseline for comparison.
Just simply run the following command:
```bash
python src/main.py --pipline direct --model qwen3-14b_vllm --dataset liar --output_dir ./output
```

## 🔥🔥🔥 0715
We defined the core concept of using `configuration files` to control the entire project, including LLM, Dataset, and Pipeline.  
For model, please refer to the `src/config/llm_config.py`.  
For dataset, please refer to the `src/config/dataset_config.py`.  
For pipline, please refer to the `src/config/pipline_config.py`.  
Now, we can instantiate the LLM as simple as:
```python
from config import supported_llm
from llm import AutoLLM

llm_cfg = supported_llm['qwen3-14b_vllm']
llm = AutoLLM.build(llm_cfg)
print(llm.chat("hello"))
```
We can also instantiate the dataset as simple as:
```python
from config import supported_dataset
from dataset import AutoDataset

dataset_cfg = supported_dataset['liar']
dataset = AutoDataset.build(dataset_cfg)
```

## 🔥 0714
We create the structure of the project:
- `src/core`, the abstract class of the project including LLM, Agent, Dataset, Pipline etc.
- `src/llm`, the implementation of the LLM class including vllm, ollama, openai backend.
- `src/dataset`, the implementation of the dataset class including liar, gsm8k etc.
- `src/config`, the configuration of the project including llm, dataset, pipline etc.
- `src/pipline`, the implementation of the pipline class including direct, chain-of-thought, etc.
- `src/main.py`, the main entry of the project.


## 📊 Data Preparation
Please refer to the [docs/prepare_dataset.md](docs/prepare_dataset.md).

## 📖 Model Preparation
Please refer to the [docs/prepare_model.md](docs/prepare_model.md).

## 🏗️ ️QuickStart
Before evaluation, you need to read the [Data Preparation](#-data-preparation) and [Model Preparation](#-model-preparation) first.
> [!IMPORTANT]
> We recommend you to use the script in the `examples` directory to run the project.
```bash
git clone https://github.com/gxlover0625/prompt-optimization.git
cd prompt-optimization/examples
bash direct_pipline.sh
```
- Different piplines may have different parameters, you can change the script to run the pipline you want.  
- If you want to change the parameters, you can modify the script directly.
- After running, you will get the results in the `output/Direct_Qwen3-14B_Liar_{timestamp}/results.json`
> [!NOTE]
> There is an alternative way to run the project, which is to run the `src/main.py` directly.
```bash
git clone https://github.com/gxlover0625/prompt-optimization.git
cd prompt-optimization # !!! please make sure you are in the root directory of the project

# direct run Qwen3-14B on Liar dataset
python src/main.py --pipline direct --model qwen3-14b_vllm --dataset liar --output_dir ./output
```

## 🤝 Acknowledgements
We were inspired by the excellent open-source project [OpenCompass](https://github.com/open-compass/opencompass), which helped simplify our development. Additionally, we would like to thank the following open-source projects for their code contributions.
- [OpenCompass](https://github.com/open-compass/opencompass) ![Star](https://img.shields.io/github/stars/open-compass/opencompass.svg?style=social&label=Star), is an LLM evaluation platform, supporting a wide range of models (Llama3, Mistral, InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets.
- [LMOps](https://github.com/microsoft/LMOps) ![Star](https://img.shields.io/github/stars/microsoft/LMOps.svg?style=social&label=Star), general technology for enabling AI capabilities w/ LLMs and MLLMs