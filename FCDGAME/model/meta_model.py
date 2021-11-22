# import torch.nn as nn
# import torch
# import pdb
# import torch.nn.functional as F
# from .ae.lstm_ae import LSTM_Embedding
# from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# from .traj_embedding import TrajEmbedding

# class Meta_model(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.trajembedding = TrajEmbedding
#         self.fea_crosse = nn.Linear(output_of_transformer_size,class_num)

#     def forward(self,data,filename_of_event,traj_label,event_len):
#         data,label = self.trajembedding(data,filename_of_event,traj_label,event_len)
#         # data_64 = []
#         # for data_1 in data:
#         #     print(len(data_1))
#         #     data_result = torch.sum(torch.stack(data_1,0),dim=0)/len(data_1) 
#         #     data_64.append(data_result)
            
#         # print(torch.stack(data_64,0).shape)
#         #torch.tensor(data_64)
#         return self.fea_crosse(data),label