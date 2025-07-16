supported_pipline = {
    "direct": {
        "pipline": "Direct",
        "output_dir": "output",
        "pipline_class": "pipline.direct.DirectPipline",
        "execution_agent": None,
        "evaluation_agent": None
    },
    "zerocot":{
        "pipline": "ZeroCoT",
        "output_dir": "output",
        "pipline_class": "pipline.zerocot.ZeroCoTPipline",
        "execution_agent": None,
        "evaluation_agent": None,
        "default_prompt": " Let's think step by step."
    }
}