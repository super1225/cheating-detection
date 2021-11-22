import torch.nn as nn


class EventEmbedding(nn.Embedding):
    def __init__(self, event_num, embed_size=512):
        super().__init__(event_num, embed_size, padding_idx=0)
