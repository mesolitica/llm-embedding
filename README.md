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

1. 600M, https://huggingface.co/mesolitica/llama2-embedding-600m-8k
2. 1B, https://huggingface.co/mesolitica/llama2-embedding-1b-8k

## Prerequisites

1. Install libraries,

```bash
pip3 install -r requirements.txt
```

### Flash Attention 2

1. Install dependencies,

```bash
pip3 install flash-attn --no-build-isolation
pip3 install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

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
bash run-600m.sh
```

- 1B,

```bash
bash run-1b.sh
```

**Each trained on SPOT Standard_NC96ads_A100_v4 in AKS**.

## Contribution

1. Thanks to https://github.com/aisyahrzk for preparing finetuning script and mining embedding dataset.
2. Thanks to https://github.com/KamarulAdha for mining embedding dataset.