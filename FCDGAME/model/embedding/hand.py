import torch.nn as nn


class HandEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(5, embed_size, padding_idx=0)
