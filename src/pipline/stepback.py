from typing import Dict
from pipline.direct import DirectPipline

class StepBackPipline(DirectPipline):
    pass
    # def build_prompt(self, example: Dict):
    #     prompt = super().build_prompt(example)
    #     if isinstance(prompt, str):
    #         prompt = prompt + self.cfg["default_prompt"]
    #     elif isinstance(prompt, list):
    #         last_message = prompt[-1]
    #         assert last_message['role'] == "user", "The last message should be a user message"
    #         last_message['content'] = last_message['content'] + self.cfg["default_prompt"]
    #         prompt[-1] = last_message
    #     return prompt