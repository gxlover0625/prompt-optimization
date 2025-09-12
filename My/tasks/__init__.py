from .base import DataLoader

def load_task(task_name: str):
    if task_name.lower() == "gsm8k":
        from .gsm8k import GSM8KBuilder, is_equal
        cfg = {
            "data_path": [
                "../data/gsm8k/main/train-00000-of-00001.parquet",
                "../data/gsm8k/main/test-00000-of-00001.parquet",
            ],
            "train_size": 200,
            "val_size": 300,
            "test_size": None
        }
        trainset, valset, testset = GSM8KBuilder.build(cfg)
        eval_fn = is_equal
        return trainset, valset, testset, eval_fn