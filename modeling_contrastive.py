
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

COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)

@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None

class LlamaModelEmbedding(LlamaModel):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.dense_layer = nn.Linear(self.config.hidden_size,1536)
        
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
                passage: Dict[str, Tensor] = None, labels = None, margin = 0.5):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        loss = None
        if labels is not None:
            distances = COSINE_DISTANCE(q_reps, p_reps)
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(margin - distances).pow(2))
            loss = losses.mean()

        return EncoderOutput(
            loss=loss,
        )

