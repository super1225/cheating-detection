import torch.nn as nn
import torch
import pdb
from .position import PositionalEmbedding
from .grid import GridEmbedding
from .timestamp import TimeEmbedding
from .hand import HandEmbedding
from .event import EventEmbedding
from .segment import SegmentEmbedding
from .angle import AngleEmbedding
from .disp import DispEmbedding

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. GridEmbedding: add position information
        2. TimeEmbedding: add time information
        3. EventEmbedding : adding event information
        4. HandEmbedding: adding hand information
        5. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, config, embed_size, dropout=0.1):
        """
        :param grid_num: number of grids
        :param time_segment_num: number of time segments
        :param event_num: number of event types
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        grid_num = (config['x_grid_nums']+1)*config['y_grid_nums'] + 3
        self.grid = GridEmbedding(grid_num=grid_num+1, embed_size=embed_size)
        print(config["time_segment"]+4)
        self.timestamp = TimeEmbedding(time_segment_num=config["time_segment"]+4, embed_size=embed_size)
        print(config["event_num"])
        self.event = EventEmbedding(event_num=config["event_num"]+3, embed_size=embed_size)
        self.hand = HandEmbedding(embed_size=embed_size)
        print(self.grid.embedding_dim)
        self.position = PositionalEmbedding(d_model=self.grid.embedding_dim, max_len=config['max_len']+2)
        self.segment = SegmentEmbedding(embed_size=self.grid.embedding_dim)
        self.angle = AngleEmbedding(embed_size=int(embed_size), angle_num=config['angle_embed_num']+4)
        self.disp = DispEmbedding(embed_size=int(embed_size), disp_num=config['disp_embed_num']+4)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, data, train_mode):

        train_mode = str(train_mode)
        # pdb.set_trace()
        if len(train_mode)==1:
            if train_mode=='0' or train_mode=='1' or train_mode=='2':
                # pdb.set_trace()
                x = self.grid(data['grid']) + self.position(data['grid']) + self.timestamp(data['timestamp']#data对应具体的数据，根据具体数据取出对应的词典里面的
#embedding
                ) + self.event(data['event']) + self.hand(data['hand'])
            elif train_mode=='3':
                x = self.grid(data['grid']) + self.position(data['grid']) + self.timestamp(data['timestamp']
                        ) + self.event(data['event']) + self.hand(data['hand']) + self.segment(data['segment'])
            elif train_mode=='4':
                x = self.timestamp(data['timestamp']) + self.event(data['event']) + self.disp(data['displacement']
                        ) + self.angle(data['angle']) + self.position(data['displacement']) + self.segment(data['segment'])

            elif train_mode in ['5', '6', '7']:

                # x = torch.cat((self.disp(data['displacement']), self.angle(data['angle'])), axis=2) 
                x = self.disp(data['displacement']) + self.angle(data['angle']) + self.position(data['displacement'])
                
                # tmp1 = self.angle(data['angle']) 
                # tmp2 = torch.unsqueeze(data['displacement'], 2)
                # x = torch.cat((tmp1, tmp2), axis=2)
                # x = x + self.position(data['angle'])

        elif train_mode[0]=='9':
            x = self.position(data['grid'])
            if train_mode[1]=='1':
                x = x + self.grid(data['grid'])
            if train_mode[2]=='1':
                x = x + self.timestamp(data['timestamp'])
            if train_mode[3]=='1':
                x = x + self.event(data['event'])
            if train_mode[4]=='1':
                # pdb.set_trace()
                x = x + self.hand(data['hand'])

        return self.dropout(x)

if __name__ == '__main__':
    bertembedding = BERTEmbedding(200*120, 4, 12, 512)