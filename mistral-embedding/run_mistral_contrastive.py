import logging
import os
import random
import math
from dataclasses import dataclass, field
from pathlib import Path
from transformers import DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import AutoConfig, AutoTokenizer, AutoModel, MistralConfig
from transformers import (
    HfArgumentParser,
    set_seed,
)
from typing import Optional, Dict
from transformers import TrainingArguments
from transformers.trainer import Trainer
from mistral_contrastive import MistralModelEmbedding
from torch.utils.data import Dataset
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from sklearn.utils import shuffle
import torch
import json

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    embedding_size: int = field(
        metadata={"help": "embedding size"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )

    train_group_size: int = field(
        default=3,
        metadata={
            "help": "max group size"
        },
    )


class ListStr(Encoding):
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)

_encodings['liststr'] = ListStr

class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args, tokenizer):
        self.dataset = LocalDataset(local = args.train_data)
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        if len(self.dataset[item]['pos']) < self.args.train_group_size:
            poss = self.dataset[item]['pos']
        else:
            poss = random.sample(self.dataset[item]['pos'], self.args.train_group_size)

        if len(self.dataset[item]['neg']) < self.args.train_group_size:
            negs = self.dataset[item]['neg']
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size)

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
        q_collated.pop('token_type_ids')
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        d_collated.pop('token_type_ids')
        return {"query": q_collated, "passage": d_collated, "labels": torch.tensor(label)}

def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = MistralConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    config.embedding_size = model_args.embedding_size

    logger.info(config)
    
    # tokenizer.padding_side = "left"

    model = MistralModelEmbedding.from_pretrained(
        model_args.model_name_or_path,
        config = config,
        use_flash_attention_2 = True,
        torch_dtype=torch.bfloat16,
    )

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer
    )

    if training_args.do_train:

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

    

if __name__ == "__main__":
    main()