import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F
from .ae.lstm_ae import LSTM_Embedding
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .traj_embedding import TrajEmbedding

class Classify(nn.Module):

    def __init__(self,trajEmbedding:TrajEmbedding, output_of_transformer_size,class_num):
        super().__init__()
        self.trajembedding = trajEmbedding
        self.fea_crosse = nn.Linear(output_of_transformer_size,class_num)

    def forward(self,data,filename_of_event,traj_label,event_len):
        data,label = self.trajembedding(data,filename_of_event,traj_label,event_len)
        return self.fea_crosse(data),label