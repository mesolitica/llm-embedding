
from transformers import MistralPreTrainedModel, MistralModel, MistralConfig
from typing import Dict
from transformers.file_utils import ModelOutput
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import nn, Tensor
from dataclasses import dataclass
from torch import nn
import torch
from transformers.file_utils import ModelOutput
import torch.nn.functional as F

COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)

@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None

class MistralModelEmbedding(MistralPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.model = MistralModel(config)
        self.dense_layer = nn.Linear(
            self.config.hidden_size,
            self.config.embedding_size,
            bias=False
        )
        self.post_init()
    

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model.forward(**features,return_dict=True)
        logits = self.dense_layer(psg_out.last_hidden_state)
        input_ids = features['input_ids']
        batch_size = input_ids.shape[0]
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits


    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None, labels = None, margin = 1.0):
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