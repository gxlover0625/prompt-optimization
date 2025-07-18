import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
from typing import Dict, List, Union
import logging
from tqdm import tqdm

from config import supported_llm
from core.llm import Message
from core.agent import Agent
from core.pipline import Pipline
from dataset.auto_dataset import AutoDataset
from llm.auto_llm import AutoLLM
from ref.protegi.main import get_evaluator, get_scorer, get_task_class
from ref.protegi.optimizers import ProTeGi
from ref.protegi.predictors import BinaryPredictor
import ref.protegi.utils as protegi_utils


def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    llm = AutoLLM.build(supported_llm['qwen3-14b_vllm_non-thinking'])
    response = llm.chat(prompt)
    return [response]

protegi_utils.chatgpt = chatgpt

class ExecutionAgent(Agent):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.llm = AutoLLM.build(cfg)
    
    def execute(self, prompt:Union[str, List[Message]])->str:
        return self.llm.chat(prompt)

class EvaluationAgent(Agent):
    def __init__(self, cfg:Dict, evaluate_fn=None):
        super().__init__(cfg)
        self.evaluate_fn = evaluate_fn
    
    def evaluate(self, *args, **kwargs):
        return self.evaluate_fn(*args, **kwargs)

class OptimizationAgent(Agent):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
    
    def optimize(self):
        pass

class ProTeGiPipline(Pipline):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.build_pipline()
    
    def build_pipline(self):
        self.dataset = AutoDataset.build(self.cfg["dataset"])
        self.execution_agent = ExecutionAgent(self.cfg["execution_agent"])
        if self.cfg["evaluation_agent"] == "default":
            self.evaluation_agent = EvaluationAgent(self.cfg["evaluation_agent"], self.dataset.evaluate)
        else:
            raise NotImplementedError("Evaluation agent is not supported in protegi pipline")
            # todo: add the evaluation agent
        
        self.cfg['eval_budget'] = self.cfg['samples_per_eval'] * self.cfg['eval_rounds'] * self.cfg['eval_prompts_per_round']
        self.task = get_task_class(self.cfg['task'])(self.cfg['data_dir'], self.cfg['max_threads'])
        self.scorer = get_scorer(self.cfg['scorer'])()
        self.evaluator = get_evaluator(self.cfg['evaluator'])(self.cfg)
        self.bf_eval = get_evaluator('bf')(self.cfg)
        self.gpt4 = BinaryPredictor(self.cfg)
        self.optimizer = ProTeGi(self.cfg, self.evaluator, self.scorer, self.cfg['max_threads'], self.bf_eval)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.logger.info(f"{self.cfg['pipline']} Pipline initialized with {self.cfg}")
        self.logger.info(f"Dataset: {self.cfg['dataset']['dataset_name']}")
        self.logger.info(f"Execution Agent: {self.cfg['execution_agent']['model']}")
        if self.cfg["evaluation_agent"] == "default":
            self.logger.info(f"Evaluation Agent: default metric in dataset")

    def run(self):
        train_exs = self.dataset.split['train']
        test_exs = self.dataset.split['test']
        # train_exs = self.task.get_train_examples()
        # test_exs = self.task.get_test_examples()
        
        os.makedirs(os.path.dirname(self.cfg['out']), exist_ok=True)
        if os.path.exists(self.cfg['out']):
            os.remove(self.cfg['out'])

        with open(self.cfg['out'], 'w') as f:
            f.write(json.dumps(self.cfg, indent=4))

        # candidates = [open("src/" + fp.strip()).read() for fp in self.cfg['prompts'].split(',')] # 这一块需要修改
        # print(candidates)
        default_prompt = self.dataset.cfg['default_prompt']
        if not default_prompt.startswith("# Task"):
            default_prompt = "# Task\n" + default_prompt
        candidates = [default_prompt]

        for round in tqdm(range(self.cfg['rounds'] + 1)):
            print("STARTING ROUND ", round)
            start = time.time()

            # expand candidates
            if round > 0:
                candidates = self.optimizer.expand_candidates(candidates, self.task, self.gpt4, train_exs)

            # score candidates
            scores = self.optimizer.score_candidates(candidates, self.task, self.gpt4, train_exs)
            [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

            # select candidates
            candidates = candidates[:self.cfg['beam_size']]
            scores = scores[:self.cfg['beam_size']]

            # record candidates, estimated scores, and true scores
            with open(self.cfg['out'], 'a') as outf:
                outf.write(f"======== ROUND {round}\n")
                outf.write(f'{time.time() - start}\n')
                outf.write(f'{candidates}\n')
                outf.write(f'{scores}\n')
            metrics = []
            for candidate, score in zip(candidates, scores):
                f1, texts, labels, preds = self.task.evaluate(self.gpt4, candidate, test_exs, n=self.cfg['n_test_exs'])
                metrics.append(f1)
            with open(self.cfg['out'], 'a') as outf:  
                outf.write(f'{metrics}\n')
        
        pass

if __name__ == "__main__":
    from config import supported_pipline, supported_dataset, supported_llm

    dataset_cfg = supported_dataset['liar']
    execution_agent_cfg = supported_llm['qwen3-14b_vllm_non-thinking']
    evaluation_agent_cfg = "default"
    pipline_cfg = supported_pipline['protegi']
    pipline_cfg['dataset'] = dataset_cfg
    pipline_cfg['execution_agent'] = execution_agent_cfg
    pipline_cfg['evaluation_agent'] = evaluation_agent_cfg

    pipline = ProTeGiPipline(pipline_cfg)
    pipline.run()
    