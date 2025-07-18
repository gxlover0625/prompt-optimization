import argparse
import os
import warnings

from config import supported_llm, supported_dataset, supported_pipline
from pipline import AutoPipline
from dataset import dataset_support_llm_as_judge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipline", type=str, default="direct")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--execution_agent", type=str, default=None)
    parser.add_argument("--evaluation_agent", type=str, default=None)
    parser.add_argument("--optimization_agent", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="liar")
    parser.add_argument("--evaluation_metric", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    pipline_cfg = supported_pipline[args.pipline]
    dataset_cfg = supported_dataset[args.dataset]

    if args.model is not None:
        llm_cfg = supported_llm[args.model]
        for key, value in pipline_cfg.items():
            if key.endswith("_agent"):
                pipline_cfg[key]['llm'] = llm_cfg
    
    if args.execution_agent is not None and "execution_agent" in pipline_cfg:
        pipline_cfg["execution_agent"]['llm'] = supported_llm[args.execution_agent]
    
    if args.optimization_agent is not None and "optimization_agent" in pipline_cfg:
        pipline_cfg["optimization_agent"]['llm'] = supported_llm[args.optimization_agent]
    
    if args.evaluation_metric == "llm_judge" and not dataset_support_llm_as_judge[dataset_cfg["dataset_name"]]:
        warnings.warn(f"{dataset_cfg['dataset_name']} does not support the LLM as judge, using the default metric in dataset.")
        args.evaluation_metric = "default"

    if "evaluation_agent" in pipline_cfg:
        pipline_cfg["evaluation_agent"]["metric"] = args.evaluation_metric
        if args.evaluation_metric == "llm_judge":
            pipline_cfg["evaluation_agent"]["llm"] = supported_llm[args.evaluation_agent]
        else:
            pipline_cfg["evaluation_agent"]["llm"] = None

    pipline_cfg["output_dir"] = args.output_dir
    pipline_cfg["dataset"] = dataset_cfg
    pipline = AutoPipline.build(pipline_cfg)
    pipline.run()

if __name__ == "__main__":
    main()