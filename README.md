# llama2-embedding

Finetune Malaysian Llama2 for Malaysian context embedding task.

## Dataset Preparation

We gathered dataset of malaysian texts and generate embedding using various embedding models.

### OpenAI ada-002

Use Open AI Embedding to mine high confidence negative and positive text pair.

Dataset: https://huggingface.co/datasets/mesolitica/OpenAI-embedding-ada-002

Refer notebook [mining-openai/mining-facebook.ipynb](mining-openai/mining-facebook.ipynb) for example of text mining process to extract high confidence pair texts.

### bge-large-en

Use https://huggingface.co/BAAI/bge-large-en to mine high confidence negative and positive text pair.

Dataset: https://huggingface.co/datasets/mesolitica/bge-large-en-embedding

Refer notebook [mining-bge/mining-twitter.ipynb](mining-bge/mining-twitter.ipynb) for example of text mining process to extract high confidence pair texts.

## Published Dataset

We published paired dataset at https://huggingface.co/datasets/mesolitica/embedding-pair-mining

## Published finetuned models

1. 600M, https://huggingface.co/mesolitica/llama2-embedding-600m-16k
2. 1B, https://huggingface.co/mesolitica/llama2-embedding-1b-16k

## Finetune

Finetune replicates https://github.com/FlagOpen/FlagEmbedding finetune process.

1. Train Data Format.

Train data is in a json file with the following format:

```
{"query": str, "pos": List[str], "neg":List[str]}
```

query is the query, and pos is a list of positive texts, neg is a list of negative texts.

Combine JSONL files into 1 file, follow [notebooks/combine-embedding.ipynb](notebooks/combine-embedding.ipynb).

2. Pretrained Model.

We used,

- 600M Malaysian Llama2, https://huggingface.co/mesolitica/llama-600m-hf-32768-fpf
- 1B Malaysian Llama2, https://huggingface.co/mesolitica/llama-1b-hf-32768-fpf

3. Train.

- 600M,

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=llama2-embedding-600m python3 run.py \
--output_dir="./embedding-model-llama-600m" \
--model_name_or_path="mesolitica/llama-600m-hf-32768-fpf" \
--train_data="shuf-train-embedding.jsonl" \
--per_device_train_batch_size="6" \
--learning_rate="2e-5" \
--num_train_epochs="5" \
--max_seq_length 16384 \
--save_steps="5000" \
--save_total_limit="3" \
--do_train \
--gradient_checkpointing \
--logging_steps 1 \
--normalized True \
--temperature 0.02 \
--query_max_len 16384 \
--passage_max_len 16384 \
--train_group_size 3  \
--sentence_pooling_method="mean" \
--bf16
```

- 1B,

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=llama2-embedding-1b python3 run.py \
--output_dir="./embedding-model-llama-1b" \
--model_name_or_path="mesolitica/llama-1b-hf-32768-fpf" \
--train_data="shuf-train-embedding.jsonl" \
--per_device_train_batch_size="4" \
--learning_rate="2e-5" \
--num_train_epochs="5" \
--max_seq_length 16384 \
--save_steps="5000" \
--save_total_limit="3" \
--do_train \
--gradient_checkpointing \
--logging_steps 1 \
--normalized True \
--temperature 0.02 \
--query_max_len 16384 \
--passage_max_len 16384 \
--train_group_size 3  \
--sentence_pooling_method="mean" \
--bf16
```

## Contribution

1. Thanks to https://github.com/aisyahrzk for preparing finetuning script and mining embedding dataset.
2. Thanks to https://github.com/KamarulAdha for mining embedding dataset.