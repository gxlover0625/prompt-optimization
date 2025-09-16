from typing import Dict, List
from prompts import error_example_template

class Optimizer:
    def __init__(self, opt_client=None, mode="acc", error_cnt=3, correct_cnt=3):
        self.opt_client = opt_client
        self.mode = mode
        self.error_cnt = error_cnt
        self.correct_cnt = correct_cnt
    
    def collect_error_examples(self, examples: List[Dict]):
        if self.mode == "acc":
            error_examples = [
                example
                for example in examples if example['score'] == 0
            ]
        return error_examples
    
    def collect_correct_examples(self, examples: List[Dict]):
        if self.mode == "acc":
            correct_examples = [
                example
                for example in examples if example['score'] == 1
            ]
        return correct_examples
    
    def get_gradients(self, error_examples: List[Dict]=None, correct_examples: List[Dict]=None):
        error_str = ""
        if len(error_examples) > 0:
            error_examples = error_examples[:self.error_cnt]
            for idx, example in enumerate(error_examples):
                error_str += error_example_template.format(
                    index=idx+1,
                    inputs=example['metadata']['user_prompt'],
                    response=example['model_prediction'],
                    label=example['label'],
                )
        return error_str