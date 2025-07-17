supported_pipline = {
    "direct": {
        "pipline": "Direct",
        "output_dir": "output",
        "pipline_class": "pipline.direct.DirectPipline",
        "execution_agent": None,
        "evaluation_agent": None,
        "default_prompt": " Ensure the response concludes with the answer in the format: <answer>answer</answer>.",
        "do_postprocess": True
    },
    "zerocot":{
        "pipline": "ZeroCoT",
        "output_dir": "output",
        "pipline_class": "pipline.zerocot.ZeroCoTPipline",
        "execution_agent": None,
        "evaluation_agent": None,
        "default_prompt": " Let's think step by step. Ensure the response concludes with the answer in the format: <answer>answer</answer>.",
        "do_postprocess": True
    },
    "stepback": {
        "pipline": "StepBack",
        "output_dir": "output",
        "pipline_class": "pipline.stepback.StepBackPipline",
        "execution_agent": None,
        "evaluation_agent": None,
        "default_prompt": " Please first think about the principles involved in solving this task which could be helpful. And Then provide a solution step by step for this question. Ensure the response concludes with the answer in the format: <answer>answer</answer>.",
        "do_postprocess": True
    },
    "rephrase": {
        "pipline": "Rephrase",
        "output_dir": "output",
        "pipline_class": "pipline.rephrase.RephrasePipline",
        "execution_agent": None,
        "evaluation_agent": None,
        "default_prompt": "\nRephrase and expand the question, and respond. Ensure the response concludes with the answer in the format: <answer>answer</answer>.",
        "do_postprocess": True
    }
}