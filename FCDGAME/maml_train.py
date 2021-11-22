import  torch, os
import os
import math
import time
import torch
from torch import tensor
import torch.nn as nn
from model.Lstm_event_embedding import Lstm_event_embedding
from model.traj_embedding import TrajEmbedding
from model.ae.lstm_ae import LSTM_Embedding
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import numpy as np
from dataset.meta_dataset import DatasetTrain
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
import json
from trainer.Mamltrainer import MAML
from model.classify import Classify
def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default='save/', help="ex)output/bert.model")
    parser.add_argument("-hs", "--h_dims", type=int, default=64, help="hidden size of lstm model")
    parser.add_argument("-o", "--output_lstm_dim", type=int, default=64, help="output size of lstm model")
    parser.add_argument("-i", "--input_dim", type=int, default=12, help="input size of lstm model")
    #parser.add_argument("-l", "--layers", type=int, default=4, help="number of layers")
    parser.add_argument("-em", "--embed_size", type=int, default=64, help="size of embedding")

    parser.add_argument("-b", "--batch_size", type=int, default=1, help="number of batch_size")
    parser.add_argument("-b_m", "--main_batch_size", type=int, default=5, help="batch of meta training")
    parser.add_argument("-b_in", "--inner_batch_size", type=int, default=5, help="batch of adaption")
    parser.add_argument("-tnum", "--task_num", type=int, default=10, help="number of tasks")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="epochs of meta training")
    parser.add_argument("-inner_e", "--inner_epochs", type=int, default=200, help="epoches of adaption")
    parser.add_argument("--sampling", type=bool, default=False)

    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")

    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1], help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    # parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    # parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    parser.add_argument("--train_mode", type=int, default=0, help="0)train and test, 1)pretrain task1, 2)pretrain task2")
    parser.add_argument("--load_file", type=str, default=None)
    parser.add_argument("--grid", type=bool, default=True ,help="location to grid")
    parser.add_argument("-sn", "--segment_num", type=int, default=10, help="number of epochs")
    parser.add_argument('--n_way', type=int, help='n way', default=2)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=100)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--inner_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--ab_class', type=int, help='meta-level outer learning rate', default=3)
    #transfoemer 参数设置
    parser.add_argument('--dim_feedforward', type=int,help='the dimension of the feedforward network model (default=2048)', default=5)
    parser.add_argument('--nhead', type=float, help='the number of heads in the multiheadattention models (default=8)', default=4)
    parser.add_argument('--num_encoder_layers',type=float, help='the number of sub-encoder-layers in the encoder (default=2)', default=2)
    #parser.add_argument('--ab_class', type=int, help='meta-level outer learning rate', default=3)

    args = parser.parse_args()

    with open("./config.json") as f:
        config = json.load(f)
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # batchsz here means total episode number
    train_dataset = DatasetTrain(config, config["maml_dataset"], mode="train", task_num = args.task_num, n_way=args.n_way, k_spt=args.k_spt, k_query=args.k_qry,cls_num=args.ab_class)
    test_dataset = DatasetTrain(config, config["maml_support"], mode="test", task_num = args.task_num, n_way=args.n_way, k_spt=args.k_spt, k_query=args.k_qry,cls_num=args.ab_class)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Building BERT model")
    lstm_model = LSTM_Embedding(config, args.embed_size,args.input_dim, args.output_lstm_dim, args.h_dims, h_activ=nn.Sigmoid(),out_activ=nn.ReLU(), train_mode=0)
    lstm_event_embedding = Lstm_event_embedding(lstm_model)
    trajEmbedding = TrajEmbedding(lstm_event_embedding,args.output_lstm_dim+4,args.dim_feedforward,args.nhead,args.num_encoder_layers)
    meta_model = Classify(trajEmbedding,args.output_lstm_dim+4,args.n_way)
    maml =  MAML(meta_model,train_data_loader, inner_lr=args.inner_lr, meta_lr=args.meta_lr,n_way=args.n_way, k_spt=args.k_spt, k_query=args.k_qry,main_batch_size=args.main_batch_size,inner_batch_size=args.inner_batch_size,inner_epochs = args.inner_epochs,task_num=args.task_num,load_file=args.load_file,train_mode=args.train_mode,out_path= args.output_path)
    if args.train_mode == 0:#train
        for iteration in range(1, args.epochs):
            maml.train(iteration,train_data_loader)
            #maml.save(iteration,args.output_path)
    else:
        maml.test(1,test_data_loader)





if __name__ == '__main__':
    main()
