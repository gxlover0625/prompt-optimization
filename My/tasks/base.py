from abc import ABC, abstractmethod
from typing import Dict
from itertools import cycle
import random

class Dataset(ABC):
    def __init__(self, cfg: Dict, *args, **kwargs):
        self.cfg = cfg

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def __len__(self):
        pass

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indices)
        
        self.data_iter = cycle(self.indices)

    def get_batch(self):
        batch_indices = [next(self.data_iter) for _ in range(self.batch_size)]
        batch_data = [self.dataset[i] for i in batch_indices]
        return batch_data
