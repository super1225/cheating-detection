import torch.nn as nn


class GridEmbedding(nn.Embedding):
    def __init__(self, grid_num, embed_size=512):
        super().__init__(grid_num, embed_size, padding_idx=0)
