import types
from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import os
import logging
import argparse
import asyncio

from dotenv import load_dotenv
from rich import print
import cohere
from openai import OpenAI

load_dotenv() # load environment variables

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run the PromptBreeder Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', type=int, default=2)     
parser.add_argument('-ts', '--num_thinking_styles', type=int, default=4)     
parser.add_argument('-e', '--num_evals', type=int, default=10)     
parser.add_argument('-n', '--simulations', type=int, default=10)     
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")       

args = vars(parser.parse_args())

total_evaluations = args['num_mutation_prompts']*args['num_thinking_styles']*args['num_evals']

# set num_workers to total_evaluations so we always have a thread 
# co = cohere.Client(api_key=os.environ['COHERE_API_KEY'],  num_workers=total_evaluations, max_retries=5, timeout=30) #override the 2 min timeout with 30s. 
def _generate(self, query:str):
    messages = [
        {
            "role": "user",
            "content": query
        }
    ]
    response = self.chat.completions.create(
        messages=messages,
        model="Qwen3-14B",
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )
    return response.choices[0].message.content

co = OpenAI(
    api_key='sk-proj-1234567890',
    base_url="http://0.0.0.0:8000/v1"
)
co.generate = types.MethodType(_generate, co)

tp_set = mutation_prompts[:int(args['num_mutation_prompts'])]
mutator_set= thinking_styles[:int(args['num_thinking_styles'])]

logger.info(f'You are prompt-optimizing for the problem: {args["problem"]}')

logger.info(f'Creating the population...')
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

logger.info(f'Generating the initial prompts...')
init_run(p, co, int(args['num_evals']))

logger.info(f'Starting the genetic algorithm...')
run_for_n(n=int(args['simulations']), population=p, model=co, num_evals=int(args['num_evals']))

print("%"*80)
print("done processing! final gen:")
print(p.units)
