import torch
from torch.nn import Module, Linear, LayerNorm

#import fine_tuning.fine_config as config
from .positional_encoding import PositionalEncoding
from .transformerencoder import TransformerEncoderLayer, TransformerEncoder
from torch.nn.init import xavier_uniform_
from .Lstm_event_embedding import Lstm_event_embedding
'''
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward(d_ff): the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
'''

class TrajEmbedding(Module):
    def __init__(self,lstm_event_embedding:Lstm_event_embedding, lstm_out_dim, d_ff, n_head, num_encoder_layers):
        super(TrajEmbedding, self).__init__()
        self.d_model = lstm_out_dim
        self.enc_embedding = lstm_event_embedding
        #self.enc_embedding = Linear(fc_up_in_dim, d_model, bias=False)
        self.enc_pos_encode = PositionalEncoding(d_model=lstm_out_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=lstm_out_dim, nhead=n_head, dim_feedforward=d_ff
        )
        encoder_norm = LayerNorm(lstm_out_dim)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self._reset_parameters()

    def forward(self, inputs,filename_of_event,traj_label,event_len):
        """
        inputs [batch_size, sequence_len, embedding_size]
        outputs [batch_size, 10]
        """
        inputs,label = self.enc_embedding(inputs,filename_of_event,traj_label,event_len)
        #return inputs,label
        #print(label)
       

        #转化为送入transformer的数据形式
        batch_num = len(inputs)
        input_padding,input_mask,length = [],[],[]
        for traj in inputs:
            length.append(len(traj))
        max_traj_len = max(length)
        embedding_dim = inputs[0][0].size(0)#每个事件对应一个64维
        #print("here",embedding_dim)
        #exit(0)
        
        #print(torch.tensor([0]*embedding_dim,dtype=torch.float32))
        PAD = torch.tensor([0]*embedding_dim,dtype=torch.float32).cuda()#未满max_length的做padding
        for traj in inputs:
            # print(max_traj_len)
            # print(len(traj))
            traj_padding =torch.stack(traj + [PAD]*(max_traj_len-len(traj)),dim=0)
            traj_mask = torch.tensor([0]*len(traj)+[1]*(max_traj_len-len(traj)),dtype=torch.bool)
            input_padding.append(traj_padding)
            input_mask.append(traj_mask)
        input_padding = torch.stack(input_padding,dim=0)
        input_mask = torch.stack(input_mask,dim=0).cuda()
        input_length = torch.tensor(length,dtype=torch.int)
    
        traj_lengths = input_length
        src_key_padding_mask = input_mask
        #print(input_padding.shape)
        inputs = torch.transpose(input_padding, 0, 1)
        #print(inputs.shape)
        # inputs [sequence_len, batch_size * sample_num, fc_in_dim]
        #print("self.d_model",self.d_model)
        inputs = self.enc_pos_encode(inputs)
        # inputs [sequence_len, batch_size * sample_num, embedding_size]
        enc_out = self.encoder(inputs, src_key_padding_mask=src_key_padding_mask)
        # enc_out [sequence_len, batch_size * sample_num, embedding_size]

        enc_out = torch.transpose(enc_out, 0, 1)
        # enc_out [batch_size * sample_num, sequence_len, embedding_size]
        #print("enc_out",enc_out.shape)
        out = torch.zeros((batch_num , self.d_model)).cuda()

        for traj_index, this_len in enumerate(traj_lengths):
            this_vec = torch.sum(enc_out[traj_index][:this_len], dim=0) / this_len
            out[traj_index] = this_vec
        return out,label

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

