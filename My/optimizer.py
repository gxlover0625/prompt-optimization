from typing import Dict, List
from prompts import error_example_template, gradient_template, revised_prompt_template

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
    
    def get_gradients(self, error_examples: List[Dict]=None, correct_examples: List[Dict]=None, cur_prompt: str=None):
        error_examples_str = ""
        if len(error_examples) > 0:
            error_examples = error_examples[:self.error_cnt]
            for idx, example in enumerate(error_examples):
                error_examples_str += error_example_template.format(
                    index=idx+1,
                    inputs=example['metadata']['user_prompt'],
                    response=example['model_prediction'],
                    label=example['label'],
                )

        gradient_meta_prompt = gradient_template.format(
            cur_prompt=cur_prompt,
            error_examples_str=error_examples_str,
        )
        gradients = self.opt_client.chat(gradient_meta_prompt)

        try:
            extracted_gradients = gradients.split("<reasons>")[1].split("</reasons>")[0]
        except:
            extracted_gradients = gradients

        return extracted_gradients
    
    def get_revised_prompts(self, error_examples: List[Dict]=None, correct_examples: List[Dict]=None, cur_prompt: str=None, gradients: str=None):
        error_examples_str = ""
        if len(error_examples) > 0:
            error_examples = error_examples[:self.error_cnt]
            for idx, example in enumerate(error_examples):
                error_examples_str += error_example_template.format(
                    index=idx+1,
                    inputs=example['metadata']['user_prompt'],
                    response=example['model_prediction'],
                    label=example['label'],
                )
        
        revised_prompt_meta_prompt = revised_prompt_template.format(
            cur_prompt=cur_prompt,
            error_examples_str=error_examples_str,
            gradients=gradients,
        )
        revised_prompt = self.opt_client.chat(revised_prompt_meta_prompt)

        try:
            extracted_revised_prompt = revised_prompt.split("<START>")[1].split("<END>")[0]
        except:
            extracted_revised_prompt = revised_prompt

        return extracted_revised_prompt
