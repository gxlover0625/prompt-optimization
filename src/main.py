import argparse
import os
import warnings

from config import supported_llm, supported_dataset, supported_pipline
from pipline.direct import DirectPipline
from dataset import dataset_support_llm_as_judge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipline", type=str, default="direct")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--execution_agent", type=str, default=None)
    parser.add_argument("--evaluation_agent", type=str, default="default")
    parser.add_argument("--optimization_agent", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="liar")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    pipline_cfg = supported_pipline[args.pipline]
    dataset_cfg = supported_dataset[args.dataset]

    if args.model is not None:
        llm_cfg = supported_llm[args.model]
        for key, value in pipline_cfg.items():
            if key.endswith("_agent"):
                pipline_cfg[key] = llm_cfg
    
    if args.execution_agent is not None and "execution_agent" in pipline_cfg:
        pipline_cfg["execution_agent"] = supported_llm[args.execution_agent]

    if args.evaluation_agent is not None and "evaluation_agent" in pipline_cfg:
        if args.evaluation_agent == "default":
            pipline_cfg["evaluation_agent"] = "default"
        elif not dataset_support_llm_as_judge[args.dataset]:
            pipline_cfg["evaluation_agent"] = "default"
            warnings.warn(f"The dataset {args.dataset} does not support the LLM as judge, using the default metric in dataset.")
        else:
            pipline_cfg["evaluation_agent"] = supported_llm[args.evaluation_agent]

    if args.optimization_agent is not None and "optimization_agent" in pipline_cfg:
        pipline_cfg["optimization_agent"] = supported_llm[args.optimization_agent]

    pipline_cfg["output_dir"] = args.output_dir
    pipline_cfg["dataset"] = dataset_cfg
    # print(pipline_cfg)


    # llm_cfg = supported_llm[args.model]
    # dataset_cfg = supported_dataset[args.dataset]
    # exp_cfg = {"output_dir": args.output_dir, "pipline": args.pipline}
    # pipline_cfg = {**llm_cfg, **dataset_cfg, **exp_cfg}
    pipline = DirectPipline(pipline_cfg)
    pipline.run()

if __name__ == "__main__":
    main()