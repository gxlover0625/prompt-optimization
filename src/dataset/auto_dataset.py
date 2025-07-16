from typing import Dict
import importlib

class AutoDataset:
    @classmethod
    def build(cls, cfg: Dict):
        dataset_name = cfg.get("dataset_name")
        if not dataset_name:
            raise ValueError("Dataset name not provided in config")
        
        try:
            # Import the dataset module dynamically
            module_path = f"dataset.{dataset_name}"
            module = importlib.import_module(module_path)
            
            # Get the dataset class from the module
            dataset_class = getattr(module, dataset_name)
            
            # Initialize and return the dataset
            return dataset_class(cfg)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load dataset '{dataset_name}': {e}")