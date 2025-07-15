import argparse
import os

from config import supported_llm, supported_dataset
from pipline.direct import DirectPipline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipline", type=str, default="direct")
    parser.add_argument("--model", type=str, default="qwen3-14b_vllm")
    parser.add_argument("--dataset", type=str, default="liar")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    llm_cfg = supported_llm[args.model]
    dataset_cfg = supported_dataset[args.dataset]
    exp_cfg = {"output_dir": args.output_dir, "pipline": args.pipline}
    pipline_cfg = {**llm_cfg, **dataset_cfg, **exp_cfg}
    pipline = DirectPipline(pipline_cfg)
    pipline.run()

if __name__ == "__main__":
    main()