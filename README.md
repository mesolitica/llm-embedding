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

## Benchmarks

We sampled test set, only take first 1000 rows for each test set, example, [test-set/test-sample-twitter.ipynb](test-set/test-sample-twitter.ipynb).

- `positive_score`, cosine score, how good similarity score for positive class.
- `negative_score`, 1 - cosine score, how good similarity score for negative class.
- `top1`, top 1 accuracy classification.
- `top3`, top 3 accuracy classification.
- `top5`, top 5 accuracy classification.
- `top10`, top 10 accuracy classification.


### OpenAI

```python
{
    'b.cari.com.my': {
        'positive score': 0.8729225971201091,
        'negative score': 0.27480777421889363,
        'top1': 0.31621790857858484,
        'top3': 0.6242955541640576,
        'top5': 0.6944270507201001,
        'top10': 0.7623669380087664,
    },
    'c.cari.com.my': {
        'positive score': 0.8173745331635356,
        'negative score': 0.3100609159718768,
        'top1': 0.08380430943129637,
        'top3': 0.21388202048746027,
        'top5': 0.27861179795125396,
        'top10': 0.3589720946661957,
    },
    'malay-news': {
        'positive score': 0.8448714707337686,
        'negative score': 0.2741472719191583,
        'top1': 0.1386895659334196,
        'top3': 0.2952593812492648,
        'top5': 0.3745441712739678,
        'top10': 0.4754734737089754,
    },
    'twitter': {
        'positive score': 0.8928321128367129,
        'negative score': 0.26488808270585834,
        'top1': 0.22942090082094518,
        'top3': 0.4919014865764367,
        'top5': 0.5930774351009541,
        'top10': 0.7248724206789439,
    },
}
```

### Llama2 Embedding 600M

```python
{
    'b.cari.com.my': {
        'positive score': 0.79568475,
        'negative score': 0.6981619672232329,
        'top1': 0.3168440826549781,
        'top3': 0.6881653099561679,
        'top5': 0.7789605510331872,
        'top10': 0.8453350031308704,
    },
    'c.cari.com.my': {
        'positive score': 0.71944785,
        'negative score': 0.7663808533028701,
        'top1': 0.08327446132108796,
        'top3': 0.18730130695867184,
        'top5': 0.23975626986930412,
        'top10': 0.3140233133168492,
    },
    'malay-news': {
        'positive score': 0.71082395,
        'negative score': 0.7160432709481884,
        'top1': 0.14268909540054112,
        'top3': 0.27584990001176335,
        'top5': 0.3640748147276791,
        'top10': 0.47112104458299026,
    },
    'twitter': {
        'positive score': 0.8202477,
        'negative score': 0.7034184992996264,
        'top1': 0.23496782782338585,
        'top3': 0.5200798757488352,
        'top5': 0.6416685156423342,
        'top10': 0.785888617705791,
    },
}
```

### Llama2 Embedding 1B

```python
{
    'b.cari.com.my': {
        'positive score': 0.82174283,
        'negative score': 0.7068604469821633,
        'top1': 0.32623669380087666,
        'top3': 0.6947401377582968,
        'top5': 0.7902316844082655,
        'top10': 0.8603631809643081,
    },
    'c.cari.com.my': {
        'positive score': 0.74685395,
        'negative score': 0.7647963317331168,
        'top1': 0.08689509007417874,
        'top3': 0.19966442953020133,
        'top5': 0.26086188625927237,
        'top10': 0.3430766513599435,
    },
    'malay-news': {
        'positive score': 0.7159956,
        'negative score': 0.776366058746266,
        'top1': 0.14610045876955652,
        'top3': 0.285260557581461,
        'top5': 0.3640748147276791,
        'top10': 0.4775908716621574,
    },
    'twitter': {
        'positive score': 0.8326124,
        'negative score': 0.748791925041112,
        'top1': 0.23053028622143332,
        'top3': 0.5342800088750832,
        'top5': 0.6498779676059463,
        'top10': 0.7925449301087197,
    },
}
```

## Post mining

We use 1B to mining back available data, scripts at [post-mining](post-mining) and published at https://huggingface.co/datasets/mesolitica/embedding-pair-mining-malaysian-llama2-1b

You can use this data to train smaller models.

## Contribution

1. Thanks to https://github.com/aisyahrzk for preparing finetuning script and mining embedding dataset.
2. Thanks to https://github.com/KamarulAdha for mining embedding dataset.