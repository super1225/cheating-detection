import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F
from .ae.lstm_ae import LSTM_Embedding
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .embedding.event import EventEmbedding


class Lstm_event_embedding(nn.Module):
    
    def __init__(self,lstm:LSTM_Embedding):
        super().__init__()
        self.lstm = lstm
        self.event = EventEmbedding(event_num=7,embed_size=4)
        #self.liner = nn.Linear(hidden_size,2)

    def forward(self, data,filename_event,label,event_len):
        hidden = self.lstm(data[:,:,:3],event_len)
        event_tensor = torch.tensor(data[:,0,3].add(1),dtype=torch.long)
        event_embedding = self.event(event_tensor.cuda())
        #print(event_embedding.shape)
        hidden_with_event = torch.cat([hidden,event_embedding],1)
        data_list = []
        for i in range(len(filename_event)):
            #print(filename_event[i][0])
            #exit()
            split_result = filename_event[i][0].split(".json_")
            filename = split_result[0]
            event_num = int(split_result[1])
            data_list.append((hidden_with_event[i],filename,event_num,label[i]))
        #print("data_list")

        data_list.sort(key=lambda x:(x[1],x[2]))
        
        #print(temp_file)
        line_traj_temp,line_traj_event_format,traj_label = [],[],[]
        temp_file = data_list[0][1]
        traj_label.append(data_list[0][3])
        for i,data_seq in enumerate(data_list):
            #print(data_list[i][1]+"_"+str(data_list[i][2]))
            #print(i)
            if data_seq[1] == temp_file:
                line_traj_temp.append(data_seq[0])
                #name_seq.append(data_seq[1]+"_"+str(data_seq[2]))
                if(i == (len(data_list)-1)):
                    line_traj_event_format.append(line_traj_temp)
                    ##print(line_traj_event_format)
                    
                    #file_list.append(name_seq)
            else:
                line_traj_event_format.append(line_traj_temp)
                traj_label.append(data_seq[3])
                #file_list.append(name_seq)
                line_traj_temp = []
                #name_seq = []
                line_traj_temp.append(data_seq[0])
                #name_seq.append(data_seq[1]+"_"+str(data_seq[2]))
                temp_file =data_seq[1]
                if(i == (len(data_list)-1)):
                    line_traj_event_format.append(line_traj_temp)
        return line_traj_event_format,traj_label
       

class LstmclassifierAggragate(nn.Module):
    
    def __init__(self,lstm:LSTM_Embedding, output_size):
        super().__init__()
        self.lstm = lstm
        self.classify = Classify(output_size)

    def forward(self, data):
        hidden_list = []
        for i in range (0, len(data)):
            hidden_list.append(self.lstm(data[i]))
        hidden = torch.mean(torch.stack(hidden_list,0),0)
        return self.classify(hidden)

class Classify(nn.Module):

    def __init__(self, output_size):
        super().__init__()
        self.fea_crosse = nn.Linear(output_size,5)

    def forward(self, hidden):
        return self.fea_crosse(hidden)