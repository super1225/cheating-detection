import torch.nn as nn


class DispEmbedding(nn.Embedding):
    def __init__(self, disp_num=10003, embed_size=512):
        super().__init__(disp_num, embed_size, padding_idx=-1)
