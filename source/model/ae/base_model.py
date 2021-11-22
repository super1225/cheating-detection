from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence, pad_sequence

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
       
        combined = torch.cat((input, hidden), 1)

        hidden = self.i2h(combined)  
        output = self.i2o(combined) 
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()
        self.num_layers = 2
        #print(self.num_layers)
        #self.layers = nn.ModuleList()
        self.layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=h_dims,
            num_layers=1,
            batch_first=True
        )

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
       
        #exit() if h is None:
        x, (h, c) = self.layer(x)
        #print("type(out),self.out_activ",out)
        #exit()
        out = self.out_activ(h).squeeze()
        return out
        # return self.out_activ(h_n).squeeze()
       
