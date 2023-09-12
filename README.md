# llama2-embedding


## Dataset Preparation

We gathered dataset of malaysian textsx and generate embedding using Open AI Embedding to mine high confidence negative and positive text pair for finetuning llama2 embedding model.

Dataset: https://huggingface.co/datasets/mesolitica/OpenAI-embedding-ada-002

Refer notebook [mining-facebook.ipynb](/home/ubuntu/embedding/mining) for example of text mining process to extract high confidence pair texts.

## Finetune

Finetune replicates https://github.com/FlagOpen/FlagEmbedding finetune process.


1. Train Data Format

Train data is in a json file with the following format:
```
{"query": str, "pos": List[str], "neg":List[str]}
```
query is the query, and pos is a list of positive texts, neg is a list of negative texts.

2. Pretrained Model

We used llama2 pretrained model from https://huggingface.co/mesolitica/llama-7b-hf-16384-fpf and only select top 4 hidden layers from the model for finetuning llama2 embedding model

3. Train

```
WANDB_PROJECT=llama-7b-embedding python3 run.py \
--output_dir="./embedding-model-llama" \
--model_name_or_path="./llama-7b-embedding" \
--train_data="/home/ubuntu/embedding/train-dataset/twitter-train-dataset.json" \
--per_device_train_batch_size="5" \
--learning_rate="2e-5" \
--num_train_epochs="5" \
--max_seq_length 4096 \
--save_steps="500" \
--save_total_limit="3" \
--do_train \
--gradient_checkpointing \
--logging_steps 20 \
--normlized True \
--temperature 0.02 \
--query_max_len 4096 \
--passage_max_len 4096 \
--train_group_size 3  \
--sentence_pooling_method="mean" \
--bf16
```



