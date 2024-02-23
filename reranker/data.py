import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from arguments import DataArguments

from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
import json


class DatasetFixed(Dataset):
    def __init__(self, local, tokenizer, max_seq_length):
        self.dataset = LocalDataset(local=local)
        self.tokenizer = tokenizer
        self.total_len = len(self.dataset)
        self.max_seq_length = max_seq_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.dataset[idx]
        input_ids = self.tokenizer.encode_plus(data['query'], data['text'], truncation=True, max_length=self.max_seq_length)
        input_ids['labels'] = data['label']
        return input_ids 

class DataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        features = [f for f in features if f is not None]
        labels = [f['labels'] for f in features]
        onehot = torch.zeros(len(features), 2)
        for no, i in enumerate(labels):
            onehot[no, i] = 1
        input_ids = [{'input_ids': f['input_ids']} for f in features]
        
        input_ids = self.tokenizer.pad(
            input_ids,
            return_tensors = 'pt', 
        )
        input_ids['labels'] = onehot
        return input_ids
