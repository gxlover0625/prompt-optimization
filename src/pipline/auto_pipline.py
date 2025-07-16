from typing import Dict
import importlib

class AutoPipline:
    @classmethod
    def build(cls, cfg:Dict):
        pipline_class = cfg["pipline_class"]
        
        # Dynamically load the class
        module_path, class_name = pipline_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        pipeline_cls = getattr(module, class_name)
        
        # Create and return an instance
        return pipeline_cls(cfg)
        