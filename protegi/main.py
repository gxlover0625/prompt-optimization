import requests
import os
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import scorers
import tasks
import predictors
import optimizers
import config as config_file


def get_task_class(task_name):
    if task_name == 'ethos':
        return tasks.EthosBinaryTask
    elif task_name == 'jailbreak':
        return tasks.JailbreakBinaryTask
    elif task_name == 'liar':
        # return tasks.DefaultHFBinaryTask
        return tasks.Liar
    elif task_name == "gsm8k":
        return tasks.GSM8K
    elif task_name == 'bbh_object_counting':
        return tasks.BBHObjectCounting
    elif task_name == 'ar_sarcasm':
        return tasks.DefaultHFBinaryTask
    else:
        raise Exception(f'Unsupported task: {task_name}')


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')



def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    elif scorer == 'll':
        return scorers.CachedLogLikelihoodScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipline', default="protegi")
    parser.add_argument('--dataset', default="liar")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument('--model', default=None, type=str, required=True)

    parser.add_argument('--task', default='ethos')
    parser.add_argument('--data_dir', default='data/ethos')
    parser.add_argument('--prompts', default='prompts/ethos.md')
    # parser.add_argument('--config', default='default.json')
    parser.add_argument('--out', default='test_out.txt')
    parser.add_argument('--max_threads', default=32, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=400, type=int)

    parser.add_argument('--minibatch_size', default=64, type=int)
    parser.add_argument('--n_gradients', default=4, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--engine', default="chatgpt", type=str)

    parser.add_argument('--evaluator', default="ucb", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    ## process the arguments
    args.task = args.dataset
    if args.task == "bbh_object_counting":
        args.data_dir = "data/bbh"
    else:
        args.data_dir = "data/" + args.task
    args.prompts = "protegi/prompts/" + args.task + ".md"
    args.out = args.output_dir + "/results.txt"

    ## modify the config file
    config_file.model = args.model
    config_file.dataset = args.dataset
    os.makedirs(args.output_dir, exist_ok=True)

    config = vars(args)
    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    task = get_task_class(args.task)(args.data_dir, args.max_threads)
    scorer = get_scorer(args.scorer)(task)
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    gpt4 = predictors.BinaryPredictor(config)

    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = [open(fp.strip()).read() for fp in args.prompts.split(',')]

    # Track the best prompt and its accuracy
    best_prompt = None
    best_acc = -1

    for round in tqdm(range(config['rounds'] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)

        # score candidates
        scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        # select candidates
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]

        # record candidates, estimated scores, and true scores
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
            outf.write(f'{time.time() - start}\n')
            outf.write(f'{candidates}\n')
            outf.write(f'{scores}\n')
        metrics = []
        for candidate, score in zip(candidates, scores):
            acc, texts, labels, preds = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs)
            metrics.append(acc)
            
            # Update best prompt if current accuracy is higher
            if acc > best_acc:
                best_acc = acc
                best_prompt = candidate
                
        with open(args.out, 'a') as outf:  
            outf.write(f'{metrics}\n')

    # Save the best prompt to the output directory
    if best_prompt is not None:
        best_prompt_path = os.path.join(args.output_dir, f"best_prompt_{args.dataset}.md")
        with open(best_prompt_path, 'w') as f:
            f.write(best_prompt)
        print(f"Best prompt saved to {best_prompt_path} (accuracy: {best_acc:.4f})")

    print("DONE!")
