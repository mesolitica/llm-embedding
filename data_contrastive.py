import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from sklearn.utils import shuffle
import json
import torch

from arguments import DataArguments

class ListStr(Encoding):
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)

_encodings['liststr'] = ListStr

class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        self.dataset = LocalDataset(local = args.train_data)
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        if len(self.dataset[item]['pos']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['pos']))
            poss = random.sample(self.dataset[item]['pos'] * num, self.args.train_group_size - 1)
        else:
            poss = random.sample(self.dataset[item]['pos'], self.args.train_group_size - 1)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)

        combine = poss + negs
        labels = [1] * len(poss) + [0] * len(negs)

        combine, labels = shuffle(combine, labels)

        return [query] * len(combine), combine, labels


@dataclass
class EmbedCollator(DataCollatorWithPadding):

    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        label = [f[2] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        if isinstance(label[0], list):
            label = sum(label, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated, "labels": torch.tensor(label)}