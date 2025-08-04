import types
from pb import create_population, init_run, run_for_n
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles

import logging
import argparse

from dotenv import load_dotenv
from rich import print
from openai import OpenAI

import config
from config import supported_llm
from llm import AutoLLM

# load_dotenv() # load environment variables

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Run the PromptBreeder Algorithm. Number of units is mp * ts.')
parser.add_argument("--pipline", type=str, default="promptbreeder")
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--execution_agent", type=str, default=None)
parser.add_argument("--evaluation_agent", type=str, default=None)
parser.add_argument("--optimization_agent", type=str, default=None)
parser.add_argument("--dataset", type=str, default="bbh_object_counting")
parser.add_argument("--evaluation_metric", type=str, default="default")
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument('-mp', '--num_mutation_prompts', type=int, default=2)     
parser.add_argument('-ts', '--num_thinking_styles', type=int, default=4)     
parser.add_argument('-e', '--num_evals', type=int, default=10)     
parser.add_argument('-n', '--simulations', type=int, default=10)     
parser.add_argument('-p', '--problem', default="Solve the math word problem, giving your answer as an arabic numeral.")       

args = parser.parse_args()
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

args = vars(args)

total_evaluations = args['num_mutation_prompts']*args['num_thinking_styles']*args['num_evals']

# set num_workers to total_evaluations so we always have a thread 
# co = cohere.Client(api_key=os.environ['COHERE_API_KEY'],  num_workers=total_evaluations, max_retries=5, timeout=30) #override the 2 min timeout with 30s. 
# def _generate(self, query:str):
#     messages = [
#         {
#             "role": "user",
#             "content": query
#         }
#     ]
#     response = self.chat.completions.create(
#         messages=messages,
#         model="Qwen3-14B",
#         extra_body={
#             "chat_template_kwargs": {
#                 "enable_thinking": False
#             }
#         }
#     )
#     return response.choices[0].message.content

# co = OpenAI(
#     api_key='sk-proj-1234567890',
#     base_url="http://0.0.0.0:8000/v1"
# )
# co.generate = types.MethodType(_generate, co)
oa_cfg = supported_llm[config.optimization_agent]
exea_cfg = supported_llm[config.execution_agent]
optimization_agent = AutoLLM.build(oa_cfg)
execution_agent = AutoLLM.build(exea_cfg)

tp_set = mutation_prompts[:int(args['num_mutation_prompts'])]
mutator_set= thinking_styles[:int(args['num_thinking_styles'])]

logger.info(f'You are prompt-optimizing for the problem: {args["problem"]}')

logger.info(f'Creating the population...')
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

logger.info(f'Generating the initial prompts...')
init_run(p, optimization_agent, execution_agent, int(args['num_evals']))

logger.info(f'Starting the genetic algorithm...')
run_for_n(n=int(args['simulations']), population=p, optimization_agent=optimization_agent, execution_agent=execution_agent, num_evals=int(args['num_evals']))

print("%"*80)
print("done processing! final gen:")
print(p.units)
