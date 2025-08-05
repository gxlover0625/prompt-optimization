import warnings
import random
import re
import logging
import os
import concurrent.futures
from typing import List

from rich import print
import time
from cohere import Client

from pb.mutation_operators import mutate
from pb import gsm
from pb.types import EvolutionUnit, Population

logger = logging.getLogger(__name__)

gsm8k_examples = gsm.read_jsonl('promptbreeder/pb/data/gsm.jsonl')

def create_population(tp_set: List, mutator_set: List, problem_description: str) -> Population:
    """samples the mutation_prompts and thinking_styles and returns a 'Population' object.

    Args:
        'size' (int): the size of the population to create.
        'problem_description (D)' (str): the problem description we are optimizing for.
    """
    data = {
        'size': len(tp_set)*len(mutator_set),
        'age': 0,
        'problem_description' : problem_description,
        'elites' : [],
        'units': [EvolutionUnit(**{
            'T' : t, 
            'M' : m,
            'P' : '',
            'fitness' : 0,
            'history' : []
            }) for t in tp_set for m in mutator_set]
    }

    return Population(**data)

def init_run(population: Population, optimization_agent, execution_agent, num_evals: int, dataset, split="train"):
    """ The first run of the population that consumes the prompt_description and 
    creates the first prompt_tasks.
    
    Args:
        population (Population): A population created by `create_population`.
    """

    start_time = time.time()

    prompts = []

    for unit in population.units:    
        template= f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        prompts.append(template)
    
    # 使用多线程加速prompt初始化
    def process_single_prompt(prompt):
        """Helper function to process a single prompt"""
        try:
            return optimization_agent.chat(prompt)
        except Exception as exc:
            logger.error(f"Exception in prompt processing: {exc}")
            return ""  # Return empty result on error
    
    # 使用 ThreadPoolExecutor 并行处理所有 prompts
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), 10)) as executor:
        # 提交所有prompts进行并行处理
        future_to_prompt = {executor.submit(process_single_prompt, prompt): i for i, prompt in enumerate(prompts)}
        
        # 初始化结果列表，保持正确的长度
        results = [""] * len(prompts)
        
        # 收集结果，保持顺序
        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt_index = future_to_prompt[future]
            try:
                result = future.result()
                results[prompt_index] = result
            except Exception as exc:
                logger.error(f"Exception getting future result: {exc}")
                results[prompt_index] = ""
    end_time = time.time()

    logger.info(f"Prompt initialization done. {end_time - start_time}s")

    assert len(results) == population.size, "size of google response to population is mismatched"
    for i, item in enumerate(results):
        population.units[i].P = item

    _evaluate_fitness(population, execution_agent, num_evals, dataset, split)
    
    return population

def run_for_n(n: int, population: Population, optimization_agent, execution_agent, num_evals: int, dataset, split="train"):
    """ Runs the genetic algorithm for n generations.
    """     
    p = population
    for i in range(n):  
        print(f"================== Population {i} ================== ")
        mutate(p, optimization_agent, dataset, split)
        print("done mutation")
        _evaluate_fitness(p, execution_agent, num_evals, dataset, split)
        print("done evaluation")

    return p

def _evaluate_fitness(population: Population, model, num_evals: int, dataset, split="train") -> Population:
    """ Evaluates each prompt P on a batch of Q&A samples, and populates the fitness values.
    """
    # need to query each prompt, and extract the answer. hardcoded 4 examples for now.
    
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    #batch = random.sample(gsm8k_examples, num_evals)
    # instead of random, its better for reproducibility 
    if split == "train":
        batch = random.sample(list(dataset.train_data), num_evals)
    else:
        batch = random.sample(list(dataset.test_data), num_evals)

    elite_fitness = -1
    examples = []
    for unit in population.units:
        # set the fitness to zero from past run.
        unit.fitness = 0
        # todo. model.batch this or multithread
        examples.append([unit.P + ' \n' + example[dataset.cfg["input_key"]] for example in batch])

    # Flatten all prompts into a single list while tracking their original positions
    all_prompts = []
    prompt_positions = []  # (unit_index, prompt_index) for each prompt
    
    for unit_index, example_batch in enumerate(examples):
        for prompt_index, prompt in enumerate(example_batch):
            all_prompts.append(prompt)
            prompt_positions.append((unit_index, prompt_index))
    
    def evaluate_single_prompt(prompt):
        """Helper function to evaluate a single prompt"""
        try:
            response = model.chat(prompt)
            return response
        except Exception as exc:
            print(f"Exception in single prompt evaluation: {exc}")
            return ""  # Return empty result on error
    
    # Initialize results structure with correct dimensions
    results = [[""] * len(example_batch) for example_batch in examples]
    
    # Use a single ThreadPoolExecutor to process all prompts in parallel
    max_workers = min(len(all_prompts), 16)  # Adjust max_workers as needed
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all prompts for parallel processing
        future_to_position = {executor.submit(evaluate_single_prompt, prompt): i 
                             for i, prompt in enumerate(all_prompts)}
        
        # Collect results as they complete, maintaining order by position
        for future in concurrent.futures.as_completed(future_to_position):
            prompt_flat_index = future_to_position[future]
            unit_index, prompt_index = prompt_positions[prompt_flat_index]
            try:
                result = future.result()
                results[unit_index][prompt_index] = result
            except Exception as exc:
                print(f"Exception getting future result: {exc}")
                results[unit_index][prompt_index] = ""


    # https://arxiv.org/pdf/2309.16797.pdf#page=5, P is a task-prompt to condition 
    # the LLM before further input Q.
    for unit_index, fitness_results in enumerate(results):
        for i, response_text in enumerate(fitness_results):
            # valid = re.search(gsm.gsm_extract_answer(batch[i]['answer']), response_text)
            valid = dataset.evaluate(response_text, batch[i][dataset.cfg["label_key"]])
            if valid:
                # 0.25 = 1 / 4 examples
                population.units[unit_index].fitness += (1 / num_evals)

            if population.units[unit_index].fitness > elite_fitness:
                # I am copying this bc I don't know how it might get manipulated by future mutations.

                # unit = population.units[unit_index]
                
                current_elite = population.units[unit_index].model_copy()
                elite_fitness = population.units[unit_index].fitness
    
    # append best unit of generation to the elites list.
    population.elites.append(current_elite)
    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")

    return population