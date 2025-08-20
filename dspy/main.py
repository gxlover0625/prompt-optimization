import argparse
import dspy
from dataset import AutoDataset
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/Qwen3-14B", api_key="sk-proj-1234567890", api_base="http://localhost:8000/v1")
dspy.configure(lm=lm)

class SimpleQASolver(dspy.Module):
    def __init__(self):
        self.solver = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.solver(question=question)

def create_metric(dataset):
    def evaluate_fn(example, pred, trace=None):
        model_prediction = pred.answer
        true_label = example.answer
        result = dataset.evaluate(model_prediction, true_label)
        return float(result)
    return evaluate_fn

def main():
    parser = argparse.ArgumentParser()

    cfg = {
        "dataset_name": "GSM8K",
        "data_path": [
            "data/gsm8k/main/train-00000-of-00001.parquet",
            "data/gsm8k/main/test-00000-of-00001.parquet",
        ],
        "label_key": "answer",
        "input_key": "question",
        "default_prompt": "Question: {question}\nAnswer:"
    }
    dataset = AutoDataset.build(cfg)
    trainset = [
        dspy.Example(
            question=example[dataset.cfg['input_key']],
            answer=example[dataset.cfg['label_key']]
        ).with_inputs('question')
        for example in dataset.train_data
    ]
    testset = [
        dspy.Example(
            question=example[dataset.cfg['input_key']],
            answer=example[dataset.cfg['label_key']]
        ).with_inputs('question')
        for example in dataset.test_data
    ]
    
    solver = SimpleQASolver()
    metric = create_metric(dataset)
    evaluator = Evaluate(
        devset=testset[:5],
        metric=metric,
        num_threads=6,
        display_progress=True,
        display_table=2
    )
    result = evaluator(solver)
    # print(result.score)
    tp = dspy.MIPROv2(
        metric=metric,
        auto="medium",
        num_threads=6
    )
    optimized_solver = tp.compile(
        solver, trainset=trainset[:30],
        max_bootstrapped_demos=2, max_labeled_demos=2
    )
    optimized_solver.save("optimized_solver.json")
    

if __name__ == "__main__":
    main()
