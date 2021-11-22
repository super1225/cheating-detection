import os
from numpy.core.fromnumeric import size
import torch
import json
import csv
import pdb
from torch.utils import data
from torch.utils.data import Dataset
import numpy as np
import random
import math
import copy
from .utils import append_angle_displacement
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from model.embedding.gener_embedding import Gener_embedding
with open("config.json") as f:
    config = json.load(f)


class DatasetTrain(Dataset): 
    """
    DataLoader read trajectory files and preprocess data, and following
    information are provided:
        1. grid information of the current points
        2. time interval between the current point and the previous point
        3. event type
        4. left or right hands
    """

    def __init__(self, config, file_directory, mode, task_num, n_way, k_spt, k_query,cls_num,p_replace=0.1, len_threshold=20, train=False, grid=True):
        """
        :param file_directory: directory path of the dataset to be loaded
        :param max_length: maximum number of points in one trajectory
        :param segment:
            1.if True, then truncate the trajectory longer than max_length into segments
            2.otherwise False, then just drop points exceeding the max_length
        """
        self.config = config
        self.grid = grid
        self.class_lable = []
        self.data_support=[]
        self.data_query=[]
        self.file_directory = file_directory
        self.max_length = config['max_len']
        self.p_replace = p_replace
        self.len_threshold = len_threshold
        self.task_num = task_num  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_spt = k_spt  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_spt  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.cls_num = cls_num#异常样本类别
        self.create_task_dataset(self.task_num,mode)

    def __len__(self):
        return self.task_num
    
    
    def to_grids(self, max_x, max_y, point):
        """
        :param max_width: the maximum value of x-axis coordinate
        :param max_height: the maximum value of y-axis coordinate
        :param r_x: the width of each grid
        :param r_y: the height of each grid
        """
        return self.config['y_grid_nums']*int(self.config['x_grid_nums']*point[0]/max_x)+int(self.config['y_grid_nums']*point[1]/max_y)

    def create_task_dataset(self, task_num, mode):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        
        self.data = []
        self.support_x_dataset = []  # support set batch
        self.query_x_dataset = []  # query set batch
        self.label = [] #lable of task
        self.mode = mode
        #print(mode)
        if mode == 'train':
            class_dirs = os.listdir(self.file_directory)
            for i in class_dirs:
                #print(i)
                class_dir = self.file_directory + '/' + str(i)
                traj_ids = os.listdir(class_dir)
                self.data.append(traj_ids)
            for b in range(task_num):  
                seq  = [int(class_dir) for class_dir in class_dirs] 
                #print("seq",seq)
                data_class_map = {value:index for index, value in enumerate(seq)}
                #print(data_class_map)
                selected_cls = random.sample(seq,self.n_way)
                support_x,query_x = [],[]
                for key_map in selected_cls:
                    
                    cls = data_class_map[key_map]
                    #print(cls)
                    selected_ab_idx = np.random.choice(len(self.data[cls]), self.k_spt + self.k_query, False)
                    np.random.shuffle(selected_ab_idx)
                    indexDtrain_ab = np.array(selected_ab_idx[:self.k_spt])  # idx for Dtrain
                    indexDtest_ab = np.array(selected_ab_idx[self.k_spt:])  # idx for Dtest
                    support_x.extend(
                        np.array(self.data[cls])[indexDtrain_ab].tolist())  # get all images filename for current Dtrain
                    query_x.extend(np.array(self.data[cls])[indexDtest_ab].tolist())
                random.shuffle(support_x)
                random.shuffle(query_x)
                self.support_x_dataset.append(support_x)  # append set to current sets  
                self.query_x_dataset.append(query_x)  # append sets to current sets
                #print("selected_class:",selected_cls)
                self.label.append(selected_cls)
        elif mode == 'test':
            file_dir_support = config["maml_support"]
            file_dir_query = config["maml_query"]
            class_dirs = os.listdir(file_dir_support)
            for i in class_dirs:
                #print(i)
                class_dir_support =  file_dir_support + '/' + str(i)
                class_dir_query =  file_dir_query + '/' + str(i)
                support_traj_ids = os.listdir(class_dir_support)
                query_traj_ids = os.listdir(class_dir_query)
                self.data_support.append(support_traj_ids)
                self.data_query.append(query_traj_ids)
            for b in range(task_num):  
                seq  = [int(class_dir) for class_dir in class_dirs]
                data_class_map = {value:index for index, value in enumerate(seq)}
                #print(seq)
                seq.remove(0)
                #print(data_class_map)
                #print(seq)
                selected_cls = random.sample(seq,1)
                support_x,query_x = [],[]
                #print(selected_cls)
                for key_map in selected_cls:
                    cls = data_class_map[key_map]
                    selected_ab_support_idx = np.random.choice(len(self.data_support[cls]), self.k_spt, False)
                    np.random.shuffle(selected_ab_support_idx)
                    indexDtrain_ab = np.array(selected_ab_support_idx)  # idx for Dtrain
                    support_x.extend(
                        np.array(self.data_support[cls])[indexDtrain_ab].tolist())  # get all images filename for current Dtrain
                    query_x.extend(np.array(self.data_query[cls]).tolist())
                    
                    #sample normal 
                    nor_data_id = data_class_map[0]
                    selected_normal_support_idx = np.random.choice(len(self.data_support[nor_data_id]), self.k_spt*1, False)
                    np.random.shuffle(selected_normal_support_idx)
                    indexDtrain_normal = np.array(selected_normal_support_idx)  # idx for Dtrain
                    # print(indexDtrain_normal)
                    # indexDtest_normal = np.array(self.data_query[nor_data_id])  # idx for Dtest
                    # print(indexDtest_normal)
                    support_x.extend(np.array(self.data_support[nor_data_id])[indexDtrain_normal].tolist())
                    query_x.extend(np.array(self.data_query[nor_data_id]).tolist())
                random.shuffle(support_x)
                random.shuffle(query_x)
                self.support_x_dataset.append(support_x)  
                self.query_x_dataset.append(query_x)  # append sets to current sets
            
                self.label.append([0]+selected_cls)
    
    def data_to_eventlist(self,line):
        line_event_format,line_event_temp = [],[]
        temp_event = line[1][0][3]
        for i in range(len(line[1])):
            if(line[1][i][3] == temp_event):
                line_event_temp.append(line[1][i])
                if(i == (len(line[1])-1)):
                    line_event_format.append(line_event_temp)
            else:
                line_event_format.append(line_event_temp)
                line_event_temp = []
                line_event_temp.append(line[1][i])
                temp_event = line[1][i][3]
        return line_event_format

    def get_data(self,file_path,filename,class_map):
        while (True):  
            #print(filename)
            with open(file_path+'/'+filename[-6]+'/'+filename) as f:
                #print(filename)
                #print(class_map)
                label = class_map[int(filename[-6])]
                line = json.load(f)
                line_event_format = self.data_to_eventlist(line)
            break

        line_o = []
        for event_num in range(len(line_event_format)):
            line_o_event = append_angle_displacement([line[0],line_event_format[event_num]], disturb=True, disturb_angle=5, disturb_disp=500)
            if self.grid:
                for i in range(0,len(line_event_format[event_num])):
                    y = (line_event_format[event_num][i][1]-line[0][3])/(line[0][1]-line[0][3])
                    x = (line_event_format[event_num][i][0]-line[0][2])/(line[0][0]-line[0][2])#int(gridnum / self.config['y_grid_nums'])
                    line_o_event[i][0] =x# int(gridnum) #x
                    line_o_event[i][1] = y
                    if i > 0:
                        line_o_event[i][2] = abs(line_event_format[event_num][i][2]-line_event_format[event_num][i-1][2])//1000
                        if line_o_event[i][2]>60:
                            line_o_event[i][2] = 60
                    else:
                        line_o_event[i][2] = 0
            line_o_event1 =torch.tensor([position[0:4] for position in line_o_event],dtype=torch.float32)
            #line_o_filename_event.append(filename+"_"+str(event_num))
            line_o.append([line_o_event1,label,filename+"_"+str(event_num)])
        return line_o

    def __getitem__(self, index):  
        support_x,query_x,support_y,query_y = [],[],[],[]
        support_data,query_data = [],[]
        #print(len(self.label))
        #SSprint(index)
        class_map = {value:index for index, value in enumerate(self.label[index])}
        flatten_support_x =  np.array(self.support_x_dataset[index]).reshape(-1,1)     
        flatten_query_x =  np.array(self.query_x_dataset[index]).reshape(-1,1)
        if self.mode == 'train':
            for i in range(len(flatten_support_x)):
                data = self.get_data(config["maml_dataset"],flatten_support_x[i][0],class_map)
                support_data.extend(data)
            #print(support_data)
        #exit()

            for j in range(len(flatten_query_x)):
                data=self.get_data(config["maml_dataset"],flatten_query_x[j][0],class_map)
                query_data.extend(data)
        else:
            for i in range(len(flatten_support_x)):
                data = self.get_data(config["maml_support"],flatten_support_x[i][0],class_map)
                support_data.extend(data)
            #print(support_data)
            #exit()

            for j in range(len(flatten_query_x)):
                data=self.get_data(config["maml_query"],flatten_query_x[j][0],class_map)
                query_data.extend(data)

        support_data.sort(key=lambda x: len(x[0]), reverse=True)
        support_event_seq_len = [len(data_seq[0]) for data_seq in support_data]
        support_y = [data_seq[1] for data_seq in support_data]
        support_x = [data_seq[0] for data_seq in support_data]
        support_filename_event = [data_seq[2] for data_seq in support_data]
        support_x = pad_sequence(support_x, batch_first=True, padding_value=0)
          
        query_data.sort(key=lambda x: len(x[0]), reverse=True)
        query_event_seq_len = [len(data_seq[0]) for data_seq in query_data]
        query_y = [data_seq[1] for data_seq in query_data]
        query_x = [data_seq[0] for data_seq in query_data]
        query_filename_event = [data_seq[2] for data_seq in query_data]
        query_x = pad_sequence(query_x, batch_first=True, padding_value=0)
        return [support_x,support_y,support_event_seq_len,support_filename_event],[query_x,query_y,query_event_seq_len, query_filename_event]
 
        