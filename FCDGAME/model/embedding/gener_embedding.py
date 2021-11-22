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


class Gener_embedding(nn.Module):

    def __init__(self, config, embed_size, dropout=0.1):
        """
        :param grid_num: number of grids
        :param time_segment_num: number of time segments
        :param event_num: number of event types
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        grid_num = (config['x_grid_nums'] + 1) * config['y_grid_nums'] + 3
        self.grid = GridEmbedding(grid_num=grid_num + 1, embed_size=embed_size)
        print(config["time_segment"] + 4)
        self.timestamp = TimeEmbedding(time_segment_num=config["time_segment"] + 4, embed_size=embed_size)
        print(config["event_num"])
        self.event = EventEmbedding(event_num=config["event_num"] + 3, embed_size=embed_size)
        #self.hand = HandEmbedding(embed_size=embed_size)
        #print(self.grid.embedding_dim)
        #self.angle = AngleEmbedding(angle_num=config['angle_embed_num'] + 4,embed_size=int(embed_size))
        #self.disp = DispEmbedding(disp_num=config['disp_embed_num'] + 4, embed_size=int(embed_size))
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, data, train_mode):
        train_mode = str(train_mode)
        if train_mode == '0':
            x = self.grid(data['grid']) + self.timestamp(data['timestamp']) + self.event(data['event'])
        elif train_mode == '1':
            a=1
            #x = self.grid(data['grid']) + self.timestamp(data['timestamp']) + self.event(data['event']) + self.angle(data['angle_embed_num']) + self.disp(data['disp_embed_num'])
        return self.dropout(x)


if __name__ == '__main__':
    embedding = Gener_embedding(200 * 120, 4, 12, 512)