from tasks import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class WorldModel:
    def __init__(
        self,
        trainset=None,
        valset=None,
        testset=None,
        train_bs: int=32,
        val_bs: int=32,
        test_bs: int=32,
        shuffle: bool=True,
        task_client=None,
        opt_client=None,
        metric=None,
        *args,
        **kwargs,
    ):
        self.train_loader = DataLoader(trainset, train_bs, shuffle)
        self.val_loader = DataLoader(valset, val_bs, shuffle)
        self.test_loader = DataLoader(testset, test_bs, shuffle)

        self.task_client = task_client
        self.opt_client = opt_client

        self.metric = metric
    
    def _process_single_item(self, item_data):
        idx, inputs, label, node = item_data

        sys_prompt = node.prompt
        user_prompt = ""
        for key, value in inputs.items():
            user_prompt += f"{key}: {value}\n"
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model_prediction = self.task_client.chat(messages)
        score = self.metric(model_prediction, label)
        return {
            **inputs,
            "model_prediction": model_prediction,
            "label": label,
            "score": score,
        }

    def evaluate_node(self, node=None):
        batch = self.val_loader.get_batch()
        batch_results = [None] * len(batch)
        score_list = [None] * len(batch)

        tasks = [
            (idx, inputs, label, node)
            for idx, (inputs, label) in enumerate(batch)
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {
                executor.submit(self._process_single_item, task): task[0]
                for task in tasks
            }

            with tqdm(total=len(tasks), desc=f"Evaluating node") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    batch_results[idx] = future.result()
                    score_list[idx] = future.result()["score"]

                    completed_scores = [s for s in score_list if s is not None]
                    avg_score = sum(completed_scores) / len(completed_scores)
                    
                    pbar.set_postfix(Score=f'{avg_score:.3f}')
                    pbar.update(1)
        
        return batch_results