import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
from typing import Dict, List, Union
import logging
from tqdm import tqdm
import copy
import concurrent.futures
from liquid import Template
import numpy as np
from collections import defaultdict

from config import supported_llm
from core.agent import Agent
from core.llm import Message
from core.pipline import Pipline
from dataset.auto_dataset import AutoDataset
from llm.auto_llm import AutoLLM
from ref.protegi.main import get_evaluator, get_scorer, get_task_class
from ref.protegi.optimizers import ProTeGi
from ref.protegi.predictors import BinaryPredictor
from ref.protegi.scorers import Cached01Scorer
import ref.protegi.utils as protegi_utils

## todo，修改optimizer的_get_gradients，里面还是调用chatgpt函数了

def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    llm = AutoLLM.build(supported_llm['qwen3-14b_vllm_non-thinking'])
    response = llm.chat(prompt)
    return [response]

protegi_utils.chatgpt = chatgpt

class SingleThreadedCached01Scorer:
    """单线程版本的scorer，不使用缓存，直接计算分数"""
    
    def __init__(self):
        pass  # 不需要缓存
    
    def __call__(self, predictor, prompts, data, agg='mean', max_threads=1):
        # 存储每个prompt的得分
        scores_by_prompt = defaultdict(list)
        
        # 遍历所有prompt和样本组合
        for prompt in prompts:
            for ex in tqdm(data, desc=f'Scoring prompt: {prompt[:20]}...'):
                try:
                    # 直接调用predictor的inference方法
                    pred = predictor.inference(ex, prompt)
                    
                    # 简单比较预测和标签
                    if pred == ex['label']:
                        scores_by_prompt[prompt].append(1)
                    else:
                        scores_by_prompt[prompt].append(0)
                except Exception as e:
                    print(f"Error scoring example {ex['id']} with prompt {prompt[:20]}: {e}")
                    scores_by_prompt[prompt].append(0)
        
        # 计算每个prompt的平均得分
        if agg == 'mean':
            return [np.mean(scores_by_prompt[prompt]) if scores_by_prompt[prompt] else 0 for prompt in prompts]
        else:
            raise Exception('Unk agg: ' + agg)

from ref.protegi.tasks import DefaultHFBinaryTask
from sklearn.metrics import accuracy_score, f1_score

# ProTeGiTask.run_evaluate
## - predictor，本质是execution agent，self.gpt4
## - prompt，本质是一个候选prompt
## - test_exs，本质是测试集
## - n，本质是测试集的数量

## evaluate会调用run_evaluate，run_evaluate会调用predictor的inference方法，inference方法会调用chatgpt

class ProTeGiTask(DefaultHFBinaryTask):
    # 这个函数是可以定制化的，传参不要改，里面的逻辑是可以改的
    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        labels = []
        preds = []
        texts = []
        
        # 使用简单的for循环替代多进程
        for ex in tqdm(test_exs[:n], desc='running evaluate'):
            # 直接调用predictor的inference方法
            pred = predictor.inference(ex, prompt)
            texts.append(ex['text'])
            labels.append(self.dataset.label_postprocess(ex['label']))
            preds.append(self.dataset.model_prediction_postprocess(pred))

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds
    
    # 这里要根据不同数据集进行重写
    def stringify_prediction(self, pred):
        # return BinaryClassificationTask.categories[pred]
        return super().stringify_prediction(pred)

class ProTeGiPredictor(BinaryPredictor):
    def __init__(self, opt, execute_fn, evaluate_fn):
        self.opt = opt
        self.execute_fn = execute_fn
        self.evaluate_fn = evaluate_fn
    
    # 这个函数是可以定制化的，传参不要改，里面的逻辑是可以改的
    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = self.execute_fn(prompt)
        # pred = self.evaluate_fn(response, ex['label'])
        # response = utils.chatgpt(
        #     prompt, max_tokens=4, n=1, timeout=2, 
        #     temperature=self.opt['temperature'])[0]
        # pred = 1 if response.strip().upper().startswith('YES') else 0
        return response


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

class OptimizationAgent(Agent):
    def __init__(self, cfg:Dict):
        super().__init__(cfg)
        self.llm = AutoLLM.build(cfg['llm'])

        self.cfg['eval_budget'] = self.cfg['samples_per_eval'] * self.cfg['eval_rounds'] * self.cfg['eval_prompts_per_round']
        # 使用自定义的单线程Scorer替代默认的scorer
        self.scorer = SingleThreadedCached01Scorer()
        self.evaluator = get_evaluator(self.cfg['evaluator'])(self.cfg)
        self.bf_eval = get_evaluator('bf')(self.cfg)
        # 将自定义scorer传递给optimizer
        self.optimizer = ProTeGi(self.cfg, self.evaluator, self.scorer, self.cfg['max_threads'], self.bf_eval)

    def expand_candidates(self, *args, **kwargs):
        return self.optimizer.expand_candidates(*args, **kwargs)
    
    def score_candidates(self, *args, **kwargs):
        return self.optimizer.score_candidates(*args, **kwargs)

    def optimize(self):
        pass

class ProTeGiPipline(Pipline):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.build_pipline()
    
    def build_pipline(self):
        self.dataset = AutoDataset.build(self.cfg["dataset"])
        self.cfg["execution_agent"]["model"] = self.cfg["execution_agent"]["llm"]["model"]
        self.execution_agent = ExecutionAgent(self.cfg["execution_agent"])
        if self.cfg["evaluation_agent"]['metric'] == "default":
            self.evaluation_agent = EvaluationAgent(self.cfg["evaluation_agent"], self.dataset.evaluate)
        else:
            raise NotImplementedError("Evaluation agent is not supported in protegi pipline")
            # todo: add the evaluation agent
        
        self.optimization_agent = OptimizationAgent(self.cfg["optimization_agent"])
        
        self.task = get_task_class(self.cfg['task'])(self.cfg['data_dir'], self.cfg['max_threads'])
        
        # self.gpt4 = BinaryPredictor(self.cfg)
        self.gpt4 = ProTeGiPredictor(self.cfg, self.execution_agent.execute, self.evaluation_agent.evaluate)
        

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.logger.info(f"{self.cfg['pipline']} Pipline initialized with {self.cfg}")
        self.logger.info(f"Dataset: {self.cfg['dataset']['dataset_name']}")
        self.logger.info(f"Execution Agent: {self.cfg['execution_agent']['model']}")
        if self.cfg["evaluation_agent"]['metric'] == "default":
            self.logger.info(f"Evaluation Agent: default metric in dataset")

    # def build_prompt(self, example: Dict, candidate_prompt: str):
        

    def execute_batch(self, examples: List[Dict]):
        pass

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
        # default_prompt = self.dataset.cfg['default_prompt']
        candidates = [self.dataset.cfg['protegi_prompt']]
        
        # 创建自定义任务类，使用for循环而不是多进程
        custom_task = ProTeGiTask(self.cfg['data_dir'], self.cfg['max_threads'])
        custom_task.dataset = self.dataset # 这个可以通过重写__init__函数来实现

        for round in tqdm(range(self.cfg['rounds'] + 1)):
            print("STARTING ROUND ", round)
            start = time.time()

            # expand candidates
            if round > 0:
                candidates = self.optimization_agent.expand_candidates(candidates, custom_task, self.gpt4, train_exs)

            # score candidates
            scores = self.optimization_agent.score_candidates(candidates, custom_task, self.gpt4, train_exs)
            [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

            # select candidates
            candidates = candidates[:self.cfg['optimization_agent']['beam_size']]
            scores = scores[:self.cfg['optimization_agent']['beam_size']]

            # record candidates, estimated scores, and true scores
            with open(self.cfg['out'], 'a') as outf:
                outf.write(f"======== ROUND {round}\n")
                outf.write(f'{time.time() - start}\n')
                outf.write(f'{candidates}\n')
                outf.write(f'{scores}\n')
            # metrics = []

            metrics = []
            for candidate, score in zip(candidates, scores):
                f1, texts, labels, preds = custom_task.evaluate(self.gpt4, candidate, test_exs, n=self.cfg['n_test_exs'])
                metrics.append(f1)
            with open(self.cfg['out'], 'a') as outf:  
                outf.write(f'{metrics}\n')
        
        pass

if __name__ == "__main__":
    from config import supported_pipline, supported_dataset, supported_llm

    dataset_cfg = supported_dataset['liar']
    pipline_cfg = supported_pipline['protegi']
    pipline_cfg['n_test_exs'] = 20
    pipline_cfg['optimization_agent']['minibatch_size'] = 16
    llm_cfg = supported_llm['qwen3-14b_vllm_non-thinking']
    pipline_cfg['execution_agent']['llm'] = llm_cfg
    pipline_cfg['evaluation_agent']['llm'] = llm_cfg
    pipline_cfg['optimization_agent']['llm'] = llm_cfg
    pipline_cfg['dataset'] = dataset_cfg
    pipline = ProTeGiPipline(pipline_cfg)
    pipline.run()
    