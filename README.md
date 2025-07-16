# prompt-optimization



## 🚀 What's New
## 🔥 0714
We create the structure of the project:
- `src/core`, the abstract class of the project including LLM, Agent, Dataset, Pipline etc.


## 📊 Data Preparation
Please refer to the [docs/prepare_dataset.md](docs/prepare_dataset.md).

## 📖 Model Preparation
Please refer to the [docs/prepare_model.md](docs/prepare_model.md).

## 🏗️ ️QuickStart
Before evaluation, you need to read the [Data Preparation](#-data-preparation) and [Model Preparation](#-model-preparation) first.
```bash
git clone https://github.com/gxlover0625/prompt-optimization.git
cd prompt-optimization # please make sure you are in the root directory of the project

# direct run Qwen3-14B on Liar dataset
python src/main.py --pipline direct --model qwen3-14b_vllm --dataset liar --output_dir ./output
```
After running, you will get the results in the `output/direct_Qwen3-14B_Liar_{timestamp}/results.json`

## 🤝 Acknowledgements
We were inspired by the excellent open-source project [OpenCompass](https://github.com/open-compass/opencompass), which helped simplify our development. Additionally, we would like to thank the following open-source projects for their code contributions.
- [OpenCompass](https://github.com/open-compass/opencompass) ![Star](https://img.shields.io/github/stars/open-compass/opencompass.svg?style=social&label=Star), is an LLM evaluation platform, supporting a wide range of models (Llama3, Mistral, InternLM2,GPT-4,LLaMa2, Qwen,GLM, Claude, etc) over 100+ datasets.
- [LMOps](https://github.com/microsoft/LMOps) ![Star](https://img.shields.io/github/stars/microsoft/LMOps.svg?style=social&label=Star), general technology for enabling AI capabilities w/ LLMs and MLLMs