import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import config
import re
import os
from datasets import load_dataset


class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate(self, predictor, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass




def process_example(ex, predictor, prompt):
    pred = predictor.inference(ex, prompt)
    return ex, pred


class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, test_exs, n=100):
        dataset_cfg = config.supported_dataset[config.dataset]
        input_key = dataset_cfg["input_key"]
        label_key = dataset_cfg["label_key"]

        labels = []
        preds = []
        texts = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(process_example, ex, predictor, prompt) for ex in test_exs[:n]]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate'):
                ex, pred = future.result()
                texts.append(ex[input_key])
                labels.append(self.label_postprocess(ex[label_key]))
                preds.append(self.model_prediction_postprocess(pred))

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        return accuracy, texts, labels, preds

    def evaluate(self, predictor, prompt, test_exs, n=100):
        while True:
            try:
                acc, texts, labels, preds = self.run_evaluate(predictor, prompt, test_exs, n=n)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return acc, texts, labels, preds


class BinaryClassificationTask(ClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return BinaryClassificationTask.categories[pred]


class EthosBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[200:]]
        return exs
    
    def get_test_examples(self):
        df = pd.read_csv(self.data_dir + '/ethos_ishate_binary_shuf.csv', sep=';', header=None)
        df = df[(df[1] <= 0) | (df[1] >= 0.7)]
        exs = df.reset_index().to_dict('records')
        exs = [{'id': x['index'], 'text': x[0], 'label': 1 if x[1] > 0.4 else 0} for x in exs[:200]]
        return exs


class JailbreakBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/train.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, l in enumerate(open(self.data_dir + '/test.tsv')):
            convo, label = l.strip().split('\t')
            label = int(label)
            text = ' '.join([x['text'].strip() for x in json.loads(convo) if x['role'] == 'user'])
            exs.append({'id': i, 'text': text, 'label': label})
        return exs


class DefaultHFBinaryTask(BinaryClassificationTask):
    categories = ['No', 'Yes']

    def get_train_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/train.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
        return exs
    
    def get_test_examples(self):
        exs = []
        for i, row in enumerate(open(self.data_dir + '/test.jsonl')):
            row = json.loads(row.strip())
            exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
        return exs

def postprocess(model_prediction:str)->str:
    if config.do_postprocess:
        if "<answer>" in model_prediction:
            model_prediction = model_prediction.split("<answer>")[-1]
        if "</answer>" in model_prediction:
            model_prediction = model_prediction.split("</answer>")[0]
        return model_prediction.strip()
    else:
        return model_prediction

class Liar(DefaultHFBinaryTask):
    def label_postprocess(self, label:str):
        return label

    def model_prediction_postprocess(self, model_prediction:str):
        model_prediction = postprocess(model_prediction)
        extracted_prediction = 1 if model_prediction.strip().upper().startswith('YES') else 0
        return extracted_prediction
    
class GSM8K(DefaultHFBinaryTask):
    def __init__(self, data_dir, max_threads=1):
        super().__init__(data_dir, max_threads)
        self.load()
    
    def load(self):
        train_data_path = os.path.join(self.data_dir, "main/train-00000-of-00001.parquet")
        self.train_data = self.load_data(train_data_path, prefix="train")
        test_data_path = os.path.join(self.data_dir, "main/test-00000-of-00001.parquet")
        self.test_data = self.load_data(test_data_path, prefix="test")
        
    def load_data(self, data_path:str, prefix:str)->List[Dict]:
        dataset = load_dataset("parquet", data_files=data_path)
        # Convert dataset to a list of dictionaries and add IDs
        examples = []
        for i, example in enumerate(dataset["train"]):
            # Create a new dictionary with the example data and add ID
            example_dict = dict(example)
            example_dict["id"] = f"{prefix}-{i}"
            examples.append(example_dict)
        return examples

    def get_train_examples(self):
        return self.train_data
    
    def get_test_examples(self):
        return self.test_data

    # copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
    def label_postprocess(self, label:str):
        return label.split('#### ')[1].replace(',', '')
    
    # copy from https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/gsm8k.py
    def model_prediction_postprocess(self, model_prediction:str)->str:
        model_prediction = postprocess(model_prediction)
        text = model_prediction.split('Question:')[0]
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
        if not numbers:
            return 'NULL'
        return numbers[-1]
        
    def stringify_prediction(self, pred):
        return pred
