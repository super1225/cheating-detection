from numpy.core.fromnumeric import shape
import torch
from torch.optim import Adam
from torch.nn import parameter
from torch.nn.modules import module
import torch.optim
import torch.nn as nn
import numpy as np
import copy
import tqdm
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
import torch.nn as nn
def replace_grad(parameter_gradients, parameter_name):
    # hook function should not modify its argument
    def replace_grad_(grad):
        return parameter_gradients[parameter_name]
    # return/bind a hook function for each parameter
    return replace_grad_
class MAML:
    def __init__(self, model, dataset, inner_lr, meta_lr,n_way, k_spt, k_query,main_batch_size,inner_batch_size,inner_epochs,task_num,load_file,train_mode,out_path):
        
        # important objects
        self.dataset = dataset
        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.model = model#.to(self.device)
        self.meta_optimizer = Adam(self.model.parameters(), meta_lr)
        
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.k_spt = k_spt 
        self.k_query = k_query
        self.task_num = task_num
        self.inner_batch = inner_batch_size
        self.main_loop_batch = main_batch_size
        self.nway = n_way
        self.inner_epochs = inner_epochs
        self.plot_losses = []
        self.loss_crosse = torch.nn.CrossEntropyLoss().cuda()
        self.load_file = load_file
        self.train_mode =train_mode
        self.out_path = out_path
        self.pre = 0
        self.recall = 0
        self.f1 = 0

        if load_file != None and load_file[:5] == 'train':  
            #()
            classify_key = ['fea_crosse.weight','fea_crosse.bias']
            pretrained_dict=torch.load(out_path + load_file)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classify_key}#filter out unnecessary keys 
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            #self.model = torch.nn.DataParallel(model)
    
    def data_to_train(self,data_after_pad,data_filename_event,data_event_seq_len,data_label):
        data = []
        for i in range(0,len(data_after_pad)):
            data.append([data_after_pad[i].tolist(),data_event_seq_len[i],data_label[i],data_filename_event[i]])
        data.sort(key=lambda x: x[1], reverse=True)
        data_event_seq_len = [data_seq[1] for data_seq in data]
        data_y = [data_seq[2] for data_seq in data]
        data_x = torch.tensor([data_seq[0] for data_seq in data])
        data_filename_event = [data_seq[3] for data_seq in data]
        data_y = torch.tensor(data_y,dtype=torch.long).to(self.device)
        return data_x,data_filename_event,data_y,data_event_seq_len
     
    
    def get_data_with_filename(self,filename,data_oftask,label_oftask,event_seq_len,filename_event):
        data_of_filename,label_of_filename,event_seq_len_of_filename,filename_event_of_filename = [],[],[],[]
        filename_str = [name_temp[0] for name_temp in filename_event]
        for i,name in enumerate(filename_str):
            if(name.startswith(filename)):
                data_of_filename.append(data_oftask[i])
                label_of_filename.append(label_oftask[i])
                event_seq_len_of_filename.append(event_seq_len[i])
                filename_event_of_filename.append(filename_event[i])

        #print(filename_event_of_filename)
        return  data_of_filename,label_of_filename,event_seq_len_of_filename,filename_event_of_filename

    def inner_loop(self, iteration,taskid,support_data_oftask,query_data_oftask,support_label_oftask,query_label_oftask,support_event_seq_len,query_event_seq_len,support_filename_event,query_filename_event):
        """inner loop of MAML
           For fist order version, create a new model and copy weights from old model for simplicity
           Sample K for update fast model, Sample K for compute meta loss for a task
           copy method: load_state_dict or deep copy
        """
        # get K samples for a specific task from trainning data
        # instantiate a completely new model using deepcopy
        # if using 2nd order approximation, fuctional_forward may needed
        fast_model = copy.deepcopy(self.model)
        fast_model = fast_model.to(self.device)
        fast_optim = Adam(fast_model.parameters(), self.inner_lr)
       
        support_filename =[filename[0].split("_")[0] for filename in support_filename_event]
        support_filename_one = list(set(support_filename))
        for epoch in range(self.inner_epochs):    
            t=tp=support_loss_sum = 0
            #print("sample_num",len(support_filename_one))
            for i in range (0,(len(support_filename_one)//self.inner_batch)+1):
                if i == len(support_filename_one)//self.inner_batch:
                    #print("==")
                    support_data_batch, support_filename_event_batch,support_event_seq_len_batch,support_label_oftask_batch = [],[],[],[]
                    filename_batch = support_filename_one[(len(support_filename_one)-self.inner_batch):len(support_filename_one)]
                    for k in range(0,len(filename_batch)):
                        data_batch = self.get_data_with_filename(filename_batch[k],support_data_oftask,support_label_oftask,support_event_seq_len,support_filename_event)
                        support_data_batch.extend(data_batch[0])
                        support_filename_event_batch.extend(data_batch[3])
                        support_event_seq_len_batch.extend(data_batch[2])
                        support_label_oftask_batch.extend(data_batch[1])
                else:
                    #print("!=")
                    support_data_batch, support_filename_event_batch,support_event_seq_len_batch,support_label_oftask_batch = [],[],[],[]
                    filename_batch = support_filename_one[(self.inner_batch*i):self.inner_batch*(i+1)]
                    for k in range(0,len(filename_batch)):
                        data_batch = self.get_data_with_filename(filename_batch[k],support_data_oftask,support_label_oftask,support_event_seq_len,support_filename_event)
                        support_data_batch.extend(data_batch[0])
                        support_filename_event_batch.extend(data_batch[3])
                        support_event_seq_len_batch.extend(data_batch[2])
                        support_label_oftask_batch.extend(data_batch[1])
                support_data_into_model,support_filename_event_into_model,support_label_into_model,data_event_seq_len_into_model = self.data_to_train(support_data_batch,support_filename_event_batch,support_event_seq_len_batch,support_label_oftask_batch)
                abnormal_detection,support_label_oftraj = fast_model.forward(support_data_into_model,support_filename_event_into_model,support_label_into_model,data_event_seq_len_into_model)
                #print("sample_num",len(support_label_oftraj))
                support_label_oftraj = torch.tensor(support_label_oftraj,dtype = torch.long).to(self.device)
                loss = self.loss_crosse(abnormal_detection,support_label_oftraj)
                fast_optim.zero_grad()
                loss.backward(retain_graph=True)
                fast_optim.step()
                support_loss_sum = support_loss_sum + loss
                pred_train = abnormal_detection.argmax(dim=-1)
                t += pred_train.eq(support_label_oftraj).sum().item()
                tp += (pred_train * support_label_oftraj).sum().item()
            print("Iteration%d_task%d_innerepoch%d_batchid%d, tp=" % (iteration, taskid,epoch,i), tp, "t=%d" % t, "support_loss=%.4f" % support_loss_sum)
            if self.train_mode == 0:
                query_loss_sum,t,tp = 0,0,0
                query_data,query_filename_event,query_label,query_event_seq_len_into_model=self.data_to_train(query_data_oftask,query_filename_event,query_event_seq_len,query_label_oftask)
                abnormal_detection,query_label_oftraj = fast_model.forward(query_data,query_filename_event,query_label,query_event_seq_len_into_model)
                
                #calculate gradients
                query_label_oftraj = torch.tensor(query_label_oftraj,dtype = torch.long).to(self.device)
                query_loss_sum = self.loss_crosse(abnormal_detection,query_label_oftraj )
                fast_weights = OrderedDict(fast_model.named_parameters())
                gradients = torch.autograd.grad(query_loss_sum, fast_weights.values(),allow_unused=True)
                named_grads = {name: g for((name, _),g) in zip(fast_weights.items(),gradients)}

                #statics
                pred = abnormal_detection.argmax(dim=-1)
                t = pred.eq(query_label_oftraj).sum().item()
                tp = (pred * query_label_oftraj).sum().item()
                print("Iteration%d_task%d, tp=" % (iteration, taskid), tp, "t=%d" % t, "query_loss=%.4f" % query_loss_sum)
        if self.train_mode == 1:
                query_loss_sum,t,tp = 0,0,0
                query_data,query_filename_event,query_label,query_event_seq_len_into_model=self.data_to_train(query_data_oftask,query_filename_event,query_event_seq_len,query_label_oftask)   
                abnormal_detection,query_label_oftraj = fast_model.forward(query_data,query_filename_event,query_label,query_event_seq_len_into_model)
                #calculate gradients
                query_label_oftraj = torch.tensor(query_label_oftraj,dtype = torch.long).to(self.device)
                query_loss_sum = self.loss_crosse(abnormal_detection,query_label_oftraj )
                fast_weights = OrderedDict(fast_model.named_parameters())
                gradients = torch.autograd.grad(query_loss_sum, fast_weights.values(),allow_unused=True)
                named_grads = {name: g for((name, _),g) in zip(fast_weights.items(),gradients)}

                #statics
                pred = abnormal_detection.argmax(dim=-1)
                t = pred.eq(query_label_oftraj).sum().item()
                tp = (pred * query_label_oftraj).sum().item()
                print("Iteration%d_task%d, tp=" % (iteration, taskid), tp, "t=%d" % t, "query_loss=%.4f" % query_loss_sum)
        return query_loss_sum,named_grads
    
    def main_loop(self,iteration,data_loader):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc="Iteration%d" % iteration,
                            total=len(data_loader),
                            bar_format="{l_bar}{r_bar}")
        task_losses_sum = 0
        task_query_losses,task_gradients = [],[]
        self.meta_optimizer.zero_grad()
        self.model = self.model.to(self.device)
       
        for i, data_batch in data_iter:
             #support data
            support,query = data_batch

            #support data
            support,query = data_batch
            support_data = support[0].squeeze()
            support_label = support[1]
            support_seq_len = support[2]
            support_filename_event = support[3]
            support_label = torch.stack(support_label,dim=0).squeeze().tolist()
            support_seq_len = torch.stack(support_seq_len,dim=0).squeeze().tolist()
            
            #query data
            query_data = query[0].squeeze()
            query_label = query[1]
            query_seq_len = query[2]
            query_filename_event = query[3]
            query_label = torch.stack(query_label,dim=0).squeeze().tolist()
            query_seq_len = torch.stack(query_seq_len,dim=0).squeeze().tolist()
           
            # print("before inner")
            # print("support data",support_data)
            # print("label",support_label)
            # print("se_len",support_seq_len)
            # print("event",support_filename_event)

            #learner of task
            task_query_loss,named_grads = self.inner_loop(iteration, i , support_data,query_data,support_label,query_label,support_seq_len,query_seq_len,support_filename_event,query_filename_event)
            task_query_losses.append(task_query_loss)
            task_gradients.append(named_grads)
            task_losses_sum = task_losses_sum + task_query_loss

            #meta learner
            if i%self.main_loop_batch == 0:
                #print(task_gradients)
                sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0) for k in task_gradients[0].keys()} 
                # register hooks for each parameter in original model
                hooks = []
                for name, param in self.model.named_parameters():
                    hooks.append(
                            param.register_hook(replace_grad(sum_task_gradients, name))
                    )
                self.meta_optimizer.zero_grad()


                # Dummy pass to build computation graph
                # Replace mean task gradients for dummy gradients
                # todo: create dummy inputs
                dummy_filename_event_batch = query_filename_event
                dummy_data_batch = query_data   
                dummy_seq_len_batch = query_seq_len
                dummy_label_batch = query_label 
                dummy_data_into_model,dummy_filename_event_into_model,dummy_label_into_model,dummy_event_seq_len_into_model=self.data_to_train(dummy_data_batch,dummy_filename_event_batch,dummy_seq_len_batch,dummy_label_batch)
                #data_dummy = pack_padded_sequence(data_dummy_batch,query_seq_len,batch_first=True).to(self.device)
                #data_dummy_label =  torch.tensor(query_label,dtype=torch.long).to(self.device)
                output_dummy,out_dummy_lable = self.model.forward(dummy_data_into_model,dummy_filename_event_into_model,dummy_label_into_model,dummy_event_seq_len_into_model)
                loss = self.loss_crosse(output_dummy, torch.tensor(out_dummy_lable,dtype = torch.long).to(self.device))
                loss.backward()
                self.meta_optimizer.step()
                self.save(iteration,i,self.out_path)
                # remember to remove hooks to release memory
                for h in hooks:
                    h.remove()
               
                task_losses_sum = 0
                task_gradients = []
                self.meta_optimizer.zero_grad()
                print("meta_loss=%.4f" % loss)
            elif (i+1)==len(data_iter):
                sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0) for k in task_gradients[0].keys()} 
                # register hooks for each parameter in original model
                hooks = []
                for name, param in self.model.named_parameters():
                    hooks.append(
                            param.register_hook(replace_grad(sum_task_gradients, name))
                    )
                self.meta_optimizer.zero_grad()
                # Dummy pass to build computation graph
                # Replace mean task gradients for dummy gradients
                # todo: create dummy inputs
                dummy_filename_event_batch = query_filename_event
                dummy_data_batch = query_data   
                dummy_seq_len_batch = query_seq_len
                dummy_label_batch = query_label 
                dummy_data_into_model,dummy_filename_event_into_model,dummy_label_into_model,dummy_event_seq_len_into_model=self.data_to_train(dummy_data_batch,dummy_filename_event_batch,dummy_seq_len_batch,dummy_label_batch)
                #data_dummy = pack_padded_sequence(data_dummy_batch,query_seq_len,batch_first=True).to(self.device)
                #data_dummy_label =  torch.tensor(query_label,dtype=torch.long).to(self.device)
                output_dummy,out_dummy_lable = self.model.forward(dummy_data_into_model,dummy_filename_event_into_model,dummy_label_into_model,dummy_event_seq_len_into_model)
                loss = self.loss_crosse(output_dummy, torch.tensor(out_dummy_lable,dtype = torch.long).to(self.device))
        
                #loss = self.loss_crosse(output_dummy, data_dummy_label)
                loss.backward()
                self.meta_optimizer.step()
                self.save(iteration,i,self.out_path)
                # remember to remove hooks to release memory
                for h in hooks:
                    h.remove()
                print("meta_loss=%.4f" % loss)

            
    def train(self, iteration,data_loader):
            self.main_loop(iteration,data_loader)

    def test(self, iteration,data_loader):
        self.model = self.model.to(self.device)
        data_iter = tqdm.tqdm(enumerate(data_loader),
                            desc="Iteration%d" % iteration,
                            total=len(data_loader),
                            bar_format="{l_bar}{r_bar}")
        self.meta_optimizer.zero_grad()
        for i, data_batch in data_iter:
            support,query = data_batch
            #support data
            #support,query = data_batch
            support_data = support[0].squeeze()
            support_label = support[1]
            support_seq_len = support[2]
            support_filename_event = support[3]
            support_label = torch.stack(support_label,dim=0).squeeze().tolist()
            support_seq_len = torch.stack(support_seq_len,dim=0).squeeze().tolist()
            #print(support_filename_event)
       
            #query data
            query_data = query[0].squeeze()
            query_label = query[1]
            query_seq_len = query[2]
            query_filename_event = query[3]
            query_label = torch.stack(query_label,dim=0).squeeze().tolist()
            query_seq_len = torch.stack(query_seq_len,dim=0).squeeze().tolist()
           

            #learner of task
            task_query_loss,named_grads = self.inner_loop(iteration, i , support_data,query_data,support_label,query_label,support_seq_len,query_seq_len,support_filename_event,query_filename_event)
            #task_query_loss,named_grads = self.inner_loop(iteration, i , support_data,query_data,support_label,query_label,support_seq_len,query_seq_len)
            
        print("avg_pre",self.pre)
        print("avg_rec",self.recall)
        print("avg_f1",self.f1)

    def save(self, epoch,taskid, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if self.load_file != None:
            output_path = file_path + "fitune_train.task%d.ep%d" % (taskid, epoch)
            torch.save(self.model.state_dict(), output_path)
            print("EP:%dtask%d Model Saved on:" % (epoch,taskid) ,output_path)

        else:
            output_path = file_path + "train.task%d.ep%d" % (taskid, epoch)
            torch.save(self.model.state_dict(), output_path)
            print("EP:%dtask%d Model Saved on:" % (epoch,taskid) ,output_path)
        return output_path
