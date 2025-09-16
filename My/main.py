import random
import numpy as np

from dotenv import load_dotenv
from pathlib import Path

from tasks import load_task
from search import BeamSearch, BeamNode
from utils import generate_synonyms
from client import OpenAIClient
from world_model import WorldModel
from optimizer import Optimizer

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    set_seed()
    load_dotenv(Path(__file__).parent / ".env", override=True)
    task_prompt = 'Let\'s think step by step and output the final answer after "####".'
    expand_fn = generate_synonyms

    task_client = OpenAIClient(
        model="Qwen3-4B-Instruct-2507",
        base_url="http://localhost:8000/v1",
        api_key="sk-proj-1234567890",
    )
    opt_client = OpenAIClient(
        model="kimi-k2-250905",
    )

    task_name = "gsm8k"
    trainset, valset, testset, eval_fn = load_task(task_name)
    world_model = WorldModel(
        trainset=trainset,
        valset=valset,
        testset=testset,
        val_bs=128,
        train_bs=64,
        task_client=task_client,
        opt_client=opt_client,
        metric=eval_fn,
    )

    optimizer = Optimizer(opt_client=opt_client, mode="acc")

    search_model = BeamSearch(
        task_prompt,
        expand_fn=expand_fn,
        task_client=task_client,
        opt_client=opt_client,
        world_model=world_model,
        optimizer=optimizer,
    )
    search_model.run()