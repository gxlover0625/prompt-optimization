import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from datetime import datetime
from config import supported_llm, supported_dataset
from dataset import AutoDataset
from core.agent import Agent
from core.pipline import Pipline
from llm import backend, AutoLLM

from core.llm import Message
from typing import Dict, List, Union

class ExecutionAgent(Agent):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.llm = AutoLLM.build(cfg['llm'])
    
    def execute(self, prompt:Union[str, List[Message]])->str:
        return self.llm.chat(prompt)

class EvaluationAgent(Agent):
    def __init__(self, cfg:Dict, evaluate_fn=None):
        super().__init__(cfg)
        self.evaluate_fn = evaluate_fn
        if cfg['metric'] == "llm_judge":
            self.llm = AutoLLM.build(cfg['llm'])
    
    def evaluate(self, *args, **kwargs):
        return self.evaluate_fn(*args, **kwargs)

class DirectPipline(Pipline):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.build_pipline()
    
    def build_pipline(self):
        self.dataset = AutoDataset.build(self.cfg["dataset"])
        self.cfg["execution_agent"]["model"] = self.cfg["execution_agent"]["llm"]["model"]
        self.execution_agent = ExecutionAgent(self.cfg["execution_agent"])
        if self.cfg["evaluation_agent"]['metric'] == "default":
            self.evaluation_agent = EvaluationAgent(self.cfg["evaluation_agent"], self.dataset.evaluate)
        else:
            raise NotImplementedError("Evaluation agent is not supported in direct pipline")
            # todo: add the evaluation agent
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.logger.info(f"{self.cfg['pipline']} Pipline initialized with {self.cfg}")
        self.logger.info(f"Dataset: {self.cfg['dataset']['dataset_name']}")
        self.logger.info(f"Execution Agent: {self.cfg['execution_agent']['llm']['model']}")
        if self.cfg["evaluation_agent"]['metric'] == "default":
            self.logger.info(f"Evaluation Agent: default metric in dataset")
    
    def build_prompt(self, example:Dict):
        prompt = self.dataset.build_prompt(example)
        if isinstance(prompt, str):
            prompt = prompt + self.cfg["default_prompt"]
        elif isinstance(prompt, list):
            last_message = prompt[-1]
            assert last_message['role'] == "user", "The last message should be a user message"
            last_message['content'] = last_message['content'] + self.cfg["default_prompt"]
            prompt[-1] = last_message
        return prompt

    def execute(self, prompt:Union[str, List[Message]])->str:
        return self.execution_agent.execute(prompt)
    
    def evaluate(self, model_prediction:str, label:str):
        return self.evaluation_agent.evaluate(model_prediction, label)
    
    def optimize(self, *args, **kwargs):
        raise NotImplementedError("Optimization is not supported in direct pipline")
    
    def postprocess(self, model_prediction:str)->str:
        if self.cfg["do_postprocess"]:
            if "<answer>" in model_prediction:
                model_prediction = model_prediction.split("<answer>")[-1]
            if "</answer>" in model_prediction:
                model_prediction = model_prediction.split("</answer>")[0]
            return model_prediction.strip()
        else:
            return model_prediction
        
    def run(self):
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_dir = f"{self.cfg['output_dir']}/{self.cfg['pipline']}_{self.cfg['execution_agent']['model']}_{self.cfg['dataset']['dataset_name']}_{timestamp}/"
        self.logger.info(f"Output dir: {final_output_dir}")
        os.makedirs(final_output_dir, exist_ok=True)

        for idx, example in enumerate(self.dataset.split["test"], 1):
            prompt = self.build_prompt(example)
            model_prediction = self.execute(prompt)
            model_prediction_extracted = self.postprocess(model_prediction)
            match = self.evaluate(model_prediction_extracted, example[self.cfg['dataset']['label_key']])
            results.append({
                "idx": f"{self.cfg['pipline']}_{self.cfg['execution_agent']['model']}_{self.cfg['dataset']['dataset_name']}_{idx}",
                "prompt": prompt,
                "model_prediction": model_prediction,
                "label": example[self.cfg['dataset']['label_key']],
                "match": match,
            })
            self.logger.info(f"[{idx}/{len(self.dataset.split['test'])}], prediction: {model_prediction}, label: {example[self.cfg['dataset']['label_key']]}, match: {match}")
            with open(f"{final_output_dir}/results.json", "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        
        self.summary(final_output_dir)
    
    def summary(self, output_dir:str):
        if self.cfg["dataset"]["dataset_name"] in ["Liar", "GSM8K", "BBHObjectCounting"]:
            summary = {
                "config": self.cfg,
                "acc": None
            }
            with open(f"{output_dir}/results.json", "r") as f:
                results = json.load(f)
            correct_count = sum(1 for result in results if result["match"])
            total = len(results)
            summary["acc"] = format(round(100 * correct_count / total, 2), ".2f")

            with open(f"{output_dir}/summary.json", "w") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    llm_cfg = supported_llm["qwen3-14b_vllm"]
    # agent = ExecutionAgent(llm_cfg)
    # print(agent.llm.chat("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"+"\nput your final answer within \\boxed{}."))


    # dataset_cfg = supported_dataset["liar"]
    # dataset = Liar(dataset_cfg)
    # print(len(dataset.split["train"]),len(dataset.split["test"]))
    # print(dataset.split["train"][0])

    ## 将llm_cfg和dataset_cfg合并成pipline_cfg
    # pipline_cfg = {**llm_cfg, **dataset_cfg}
    # print(pipline_cfg)
    # pipline = DirectPipline(pipline_cfg)
    # print(pipline.execution_agent.llm.chat("你好"))

    # pipline.run()
