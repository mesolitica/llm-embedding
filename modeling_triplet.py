
from transformers import LlamaModel, LlamaConfig, LlamaTokenizer
from typing import Dict
from transformers.file_utils import ModelOutput
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import nn, Tensor
from dataclasses import dataclass
from torch import nn
from typing import Dict
import torch
from transformers.file_utils import ModelOutput
import torch.nn.functional as F

EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)

@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None

class LlamaModelEmbedding(LlamaModel):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.dense_layer = nn.Linear(self.config.hidden_size,1536)
        self.loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        
    def sentence_embedding(self, hidden_state, mask):
        if self.config.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.config.sentence_pooling_method == 'cls':
            return hidden_state[:,0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = super().forward(**features,return_dict=True)
        output = self.dense_layer(psg_out.last_hidden_state)
        p_reps = self.sentence_embedding(output, features['attention_mask'])
        if self.config.normalized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()


    def forward(self, query: Dict[str, Tensor] = None,
                pos: Dict[str, Tensor] = None, neg = None, triplet_margin = 2.0):
        rep_anchor = self.encode(query)
        rep_pos = self.encode(pos)
        n_reps = self.encode(neg)

        loss = self.loss(rep_anchor, rep_pos, n_reps)

        return EncoderOutput(
            loss=loss,
        )

