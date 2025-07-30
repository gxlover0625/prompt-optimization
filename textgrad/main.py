import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random
import os
from datetime import datetime

import json
import config
from config import supported_llm, supported_dataset
from llm import AutoLLM
from textgrad.engine.local_model_openai_api import ChatClient
from typing import Dict

load_dotenv(override=True)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def build_client(cfg: Dict):
    llm = AutoLLM.build(cfg)
    client = ChatClient(
        client=llm,
        model_string=cfg['model']
    )
    return client

def eval_sample(item, eval_fn, model):
    """
    This function allows us to evaluate if an answer to a question in the prompt is a good answer.

    """
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except Exception as e:
        print(e)
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)

def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list 

def run_test_revert(system_prompt: tg.Variable, results, model, eval_fn, test_set, max_samples=None):
    # Use test_set for prompt optimization - no validation dataset needed
    if max_samples is None:
        max_samples = len(test_set)
    test_performance = np.mean(eval_dataset(test_set, eval_fn, model, max_samples=max_samples))
    previous_performance = np.mean(results["test_acc"][-1])
    print("test_performance: ", test_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if test_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        test_performance = previous_performance

    results["test_acc"].append(test_performance)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipline", type=str, default="textgrad")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--execution_agent", type=str, default=None)
    parser.add_argument("--evaluation_agent", type=str, default=None)
    parser.add_argument("--optimization_agent", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="bbh_object_counting")
    parser.add_argument("--evaluation_metric", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    if args.dev:
        max_samples = 20

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_cfg = supported_dataset[args.dataset]
    llm_cfg = supported_llm[args.model]
    final_output_dir = f"{args.output_dir}/{args.pipline}_{llm_cfg['model']}_{dataset_cfg['dataset_name']}_{timestamp}/"
    os.makedirs(final_output_dir, exist_ok=True)
    if args.model is not None:
        config.execution_agent = args.model
        config.evaluation_agent = args.model
        config.optimization_agent = args.model
    if args.execution_agent is not None:
        config.execution_agent = args.execution_agent
    
    if args.evaluation_agent is not None and args.evaluation_metric == "llm_judge":
        config.evaluation_agent = args.evaluation_agent
    elif args.evaluation_metric == "llm_judge":
        config.evaluation_agent = args.model
    else:
        config.evaluation_agent = "default"

    if args.optimization_agent is not None:
        config.optimization_agent = args.optimization_agent

    set_seed(12)
    execution_client = build_client(supported_llm[config.execution_agent])
    optimization_client = build_client(supported_llm[config.optimization_agent])
    tg.set_backward_engine(optimization_client, override=True)

    # Load the data and the evaluation function
    train_set, _, test_set, eval_fn = load_task(args.dataset, evaluation_api=None, dataset_cfg=supported_dataset[args.dataset])
    print("Train/Test Set Lengths: ", len(train_set), len(test_set))
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
    print(STARTING_SYSTEM_PROMPT)

    train_loader = tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                                requires_grad=True,
                                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
    execution_agent = tg.BlackboxLLM(execution_client, system_prompt)
    optimization_agent = tg.TextualGradientDescent(engine=optimization_client, parameters=[system_prompt])

    results = {"test_acc": [], "prompt": []}
    results["test_acc"].append(eval_dataset(test_set, eval_fn, execution_agent, max_samples=max_samples))
    results["prompt"].append(system_prompt.get_value())

    for epoch in range(3):
        for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            pbar.set_description(f"Training step {steps}. Epoch {epoch}")
            optimization_agent.zero_grad()
            losses = []
            for (x, y) in zip(batch_x, batch_y):
                x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
                y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
                response = execution_agent(x)
                try:
                    eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
                except:
                    eval_output_variable = eval_fn([x, y, response])
                losses.append(eval_output_variable)
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimization_agent.step()
            
            run_test_revert(system_prompt, results, execution_agent, eval_fn, test_set, max_samples=max_samples)

            print("sys prompt: ", system_prompt)
            results["prompt"].append(system_prompt.get_value())
            if steps == 3:
                break
    
    summary_results = {
        "config": vars(args),
        "best_acc": float(results['test_acc'][-1]),
        "best_prompt": results['prompt'][-1]
    }
    with open(f"{final_output_dir}/results.json", "w") as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()