import argparse
import concurrent
from dotenv import load_dotenv
from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task
import numpy as np
import random

import config
from config import supported_llm, supported_dataset
from llm import AutoLLM
from textgrad.engine.local_model_openai_api import ChatClient
from typing import Dict

load_dotenv(override=True)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def build_agent(cfg: Dict):
    llm = AutoLLM.build(cfg)
    agent = ChatClient(
        client=llm,
        model_string=cfg['model']
    )
    return agent

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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

def run_test_revert(system_prompt: tg.Variable, results, model, eval_fn, test_set):
    # Use test_set for prompt optimization - no validation dataset needed
    test_performance = np.mean(eval_dataset(test_set, eval_fn, model))
    previous_performance = np.mean(results["test_acc"][-1])
    print("test_performance: ", test_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]

    if test_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        test_performance = previous_performance

    results["test_acc"].append(test_performance)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

set_seed(12)
# client_eval = OpenAI(
#     base_url="http://0.0.0.0:8000/v1",
#     api_key="eval",
# )
# client_test = OpenAI(
#     base_url="http://0.0.0.0:8000/v1",
#     api_key="test",
# )
# llm_api_eval = tg.get_engine(engine_name="gpt-4o")
# llm_api_eval = ChatExternalClient(
#     client=client_eval,
#     model_string="Qwen3-14B"
# )
# llm_api_test = tg.get_engine(engine_name="gpt-3.5-turbo-0125")
# llm_api_test = ChatExternalClient(
#     client=client_test,
#     model_string="Qwen3-14B"
# )
llm_api_eval = build_agent(supported_llm[config.execution_agent])
llm_api_test = build_agent(supported_llm[config.execution_agent])

tg.set_backward_engine(llm_api_eval, override=True)

# Load the data and the evaluation function
train_set, _, test_set, eval_fn = load_task("bbh_object_counting", evaluation_api=llm_api_eval, dataset_cfg=supported_dataset[args.dataset])
print("Train/Test Set Lengths: ", len(train_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()

print(STARTING_SYSTEM_PROMPT)

train_loader = tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)

# Testing the 0-shot performance of the evaluation engine
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True, 
                            role_description="system prompt to the language model")
model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True,
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
model = tg.BlackboxLLM(llm_api_test, system_prompt)

optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

results = {"test_acc": [], "prompt": []}
results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())

for epoch in range(3):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        optimizer.zero_grad()
        losses = []
        for (x, y) in zip(batch_x, batch_y):
            x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
            y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
            response = model(x)
            try:
                eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            except:
                eval_output_variable = eval_fn([x, y, response])
            losses.append(eval_output_variable)
        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()
        
        run_test_revert(system_prompt, results, model, eval_fn, test_set)

        print("sys prompt: ", system_prompt)
        results["prompt"].append(system_prompt.get_value())
        if steps == 3:
            break