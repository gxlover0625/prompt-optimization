import random
import numpy as np
from typing import Optional, List

from utils import generate_synonyms
from client import OpenAIClient

class BeamNode:
    # (prompt, score), depth, parent, childs
    def __init__(self, prompt: str, parent: "Optional[BeamNode]"=None):
        self.prompt = prompt
        self.score = 0.

        # process parent & childs
        self.parent = parent
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        
        self.childs: "List[BeamNode]" = []
    
    def __str__(self):
        return f"BeamNode(prompt={self.prompt}, depth={self.depth}, score={self.score})"
    
    def __repr__(self):
        return self.__str__()
    
class BeamSearch:
    def __init__(self, 
        task_prompt: str, 
        max_depth: int=5, 
        beam_size: int=3, 
        expand_fn=None,
        task_client=None,
        opt_client=None,
        world_model=None,
        optimizer=None,
        *args,
        **kwargs
    ):
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.root_node = BeamNode(task_prompt)
        self.nodes: List[BeamNode] = [self.root_node]

        self.expand_fn = expand_fn
        self.task_client = task_client
        self.opt_client = opt_client

        self.world_model = world_model
        self.optimizer = optimizer
    
    def run(self):
        root_node_results = self.world_model.evaluate_node(self.root_node)
        self.root_node.score = np.mean([result["score"] for result in root_node_results])
        
        for depth in range(self.max_depth):
            self.search()
        
    def search(self, *args, **kwargs):
        next_nodes = []
        for node in self.nodes:
            new_nodes = self.expand(node, *args, **kwargs)
            next_nodes.extend(new_nodes)
        
        for node in next_nodes:
            node_results = self.world_model.evaluate_node(node)
            node.score = np.mean([result["score"] for result in node_results])
        
        self.nodes = sorted(next_nodes, key=lambda x: x.score, reverse=True)[:self.beam_size]

    def expand(self, cur_node: BeamNode, *args, **kwargs):
        new_nodes: List[BeamNode] = []
        for _ in range(self.beam_size):
            cur_node_results = self.world_model.evaluate_node(cur_node, split="train")
            error_examples = self.optimizer.collect_error_examples(cur_node_results)
            correct_examples = self.optimizer.collect_correct_examples(cur_node_results)
            gradients = self.optimizer.get_gradients(error_examples, correct_examples, cur_node.prompt)
            new_prompt = self.optimizer.get_revised_prompts(error_examples, correct_examples, cur_node.prompt, gradients)

            new_node = BeamNode(new_prompt, cur_node)
            cur_node.childs.append(new_node) 
            
            new_nodes.append(new_node)
        return new_nodes

if __name__ == '__main__':
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent / ".env", override=True)

    random.seed(42)

    search_model = BeamSearch("Solve this question", expand_fn=generate_synonyms, task_client=OpenAIClient(model="kimi-k2-250905"), opt_client=OpenAIClient(model="kimi-k2-250905"))
    # print(search_model.root_node)
    search_model.search()