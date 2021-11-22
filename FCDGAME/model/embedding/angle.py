import torch.nn as nn


class AngleEmbedding(nn.Embedding):
    def __init__(self, angle_num=364, embed_size=512):
        super().__init__(angle_num, embed_size, padding_idx=-1)
