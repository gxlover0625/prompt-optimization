import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from config import supported_llm, supported_dataset
from dataset import AutoDataset
from core.agent import Agent
from core.pipline import Pipline
from llm import backend

from core.llm import Message
from typing import Dict, List, Union

class ExecutionAgent(Agent):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.llm = backend[cfg["backend"]](cfg)
    
    def execute(self, prompt:Union[str, List[Message]])->str:
        return self.llm.chat(prompt)

class DirectPipline(Pipline):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.build_pipline()
    
    def build_pipline(self):
        self.execution_agent = ExecutionAgent(self.cfg)
        self.dataset = AutoDataset.build_dataset(self.cfg)
    
    def run(self):
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_dir = f"{self.cfg['output_dir']}/{self.cfg['pipline']}_{self.cfg['model']}_{self.cfg['dataset_name']}_{timestamp}/"
        os.makedirs(final_output_dir, exist_ok=True)
        for idx, example in enumerate(self.dataset.split["test"], 1):
            prompt = self.dataset.build_prompt(example)
            model_prediction = self.execution_agent.execute(prompt)
            results.append({
                "idx": f"{self.cfg['model']}_{self.cfg['dataset_name']}_{idx}",
                "prompt": prompt,
                "model_prediction": model_prediction,
                "label": example[self.cfg["label_key"]],
                "match": self.dataset.evaluate(model_prediction, example[self.cfg["label_key"]]),
            })
            with open(f"{final_output_dir}/results.json", "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    llm_cfg = supported_llm["qwen3-14b_vllm"]
    # agent = ExecutionAgent(llm_cfg)
    # print(agent.llm.chat("你好"))

    dataset_cfg = supported_dataset["liar"]
    # dataset = Liar(dataset_cfg)
    # print(len(dataset.split["train"]),len(dataset.split["test"]))
    # print(dataset.split["train"][0])

    ## 将llm_cfg和dataset_cfg合并成pipline_cfg
    pipline_cfg = {**llm_cfg, **dataset_cfg}
    # print(pipline_cfg)
    pipline = DirectPipline(pipline_cfg)
    # print(pipline.execution_agent.llm.chat("你好"))

    pipline.run()
