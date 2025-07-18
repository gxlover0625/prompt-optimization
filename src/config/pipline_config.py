supported_pipline = {
    "direct": {
        "pipline": "Direct",
        "output_dir": "output",
        "pipline_class": "pipline.direct.DirectPipline",
        "execution_agent": {
            "llm": None
        },
        "evaluation_agent": {
            "llm": None,
            "metric": "default"
        },
        "default_prompt": " Ensure the response concludes with the answer in the format: <answer>answer</answer>.",
        "do_postprocess": True
    },
    "zerocot":{
        "pipline": "ZeroCoT",
        "output_dir": "output",
        "pipline_class": "pipline.zerocot.ZeroCoTPipline",
        "execution_agent": {
            "llm": None
        },
        "evaluation_agent": {
            "llm": None,
            "metric": "default"
        },
        "default_prompt": " Let's think step by step. Ensure the response concludes with the answer in the format: <answer>answer</answer>.",
        "do_postprocess": True
    },
    "stepback": {
        "pipline": "StepBack",
        "output_dir": "output",
        "pipline_class": "pipline.stepback.StepBackPipline",
        "execution_agent": {
            "llm": None
        },
        "evaluation_agent": {
            "llm": None,
            "metric": "default"
        },
        "default_prompt": " Please first think about the principles involved in solving this task which could be helpful. And Then provide a solution step by step for this question. Ensure the response concludes with the answer in the format: <answer>answer</answer>.",
        "do_postprocess": True
    },
    "rephrase": {
        "pipline": "Rephrase",
        "output_dir": "output",
        "pipline_class": "pipline.rephrase.RephrasePipline",
        "execution_agent": {
            "llm": None
        },
        "evaluation_agent": {
            "llm": None,
            "metric": "default"
        },
        "default_prompt": "\nRephrase and expand the question, and respond. Ensure the response concludes with the answer in the format: <answer>answer</answer>.",
        "do_postprocess": True
    },
    "protegi": {
        "pipline": "ProTeGi",
        "output_dir": "output/temp",
        "pipline_class": "ref.protegi.main",
        "execution_agent": {
            "llm": None
        },
        "evaluation_agent": {
            "llm": None,
            "metric": "default"
        },
        "default_prompt": "",
        "do_postprocess": True,
        "out": "output/temp/test_out.txt",
        "optimization_agent":{
            "llm": None,
            "task": "liar",
            "data_dir": "data/liar",
            "prompts": "ref/protegi/prompts/liar.md",
            "max_threads": 1,
            "temperature": 0.0,
            "optimizer": "nl-gradient",
            "rounds": 6,
            "beam_size": 4,
            "n_test_exs": 400,
            "minibatch_size": 64,
            "n_gradients": 4,
            "errors_per_gradient": 4,
            "gradients_per_error": 1,
            "steps_per_gradient": 1,
            "mc_samples_per_step": 2,
            "max_expansion_factor": 8,
            "engine": "chatgpt",
            "evaluator": "ucb",
            "scorer": "01",
            "eval_rounds": 8,
            "eval_prompts_per_round": 8,
            "samples_per_eval": 32,
            "c": 1.0,
            "knn_k": 2,
            "knn_t": 0.993,
            "reject_on_errors": False
        }
    }
}