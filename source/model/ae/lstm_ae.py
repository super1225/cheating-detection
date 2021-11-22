from torch import dtype, long
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from .base_model import Encoder
from model.embedding.gener_embedding import Gener_embedding
from model.embedding.timestamp import TimeEmbedding


class LSTM_AE(nn.Module):
    def __init__(self, input_dim, encoding_dim, h_dims=[], h_activ=nn.Sigmoid(), out_activ=nn.Tanh()):
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ,
                               out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1],
                               h_activ)

    def forward(self, x):
        # print("lstm")
        # print(x.shape())
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)

        return x


class LSTM_Embedding(nn.Module):
    def __init__(self, config,embed_size,input_dim, output_dim, h_dims, h_activ=nn.Sigmoid(),out_activ=nn.Tanh(), train_mode=0):  # （8，64，6，5，[64,128]）
        super(LSTM_Embedding, self).__init__()

        self.encoder = Encoder(input_dim, output_dim, h_dims, h_activ,out_activ)
        self.embedding = TimeEmbedding(time_segment_num=config["time_segment"]+2, embed_size=10)
        self.train_mode = train_mode
       

    def forward(self, data,event_len):
        # print(pack_padded_sequence(data, event_len, batch_first=True))
        time_tensor = data[:,:,2].add(1)
        time_embedding = self.embedding(torch.as_tensor(time_tensor,dtype=torch.long).cuda()).cuda()
        data = torch.cat([data[:,:,:2].cuda(),time_embedding],2)
        data_after_pack = pack_padded_sequence(data, event_len, batch_first=True)
        # print("data",data_after_pack)
        # exit()
       
        z = self.encoder(data_after_pack)  
        return z
