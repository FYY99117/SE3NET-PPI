import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from my_data import *
from my_model import CAN
from my_utils import *
from tensorboardX import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--description', default=None, type=str,
                    help='train description')
parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--emb_path', default=None, type=str,
                    help="The path to the pre-trained model file")
parser.add_argument('--cp_path', default=None, type=str,
                    help="Protein contact map file path")
parser.add_argument('--vec_path', default=None, type=str,
                    help='protein sequence vector path')
parser.add_argument('--split_new', default=None, type=boolean_string,
                    help='split new index file or not')
parser.add_argument('--split_mode', default=None, type=str,
                    help='split method, random, bfs or dfs')
parser.add_argument('--train_valid_index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--use_lr_scheduler', default=None, type=boolean_string,
                    help="train use learning rate scheduler or not")
parser.add_argument('--save_path', default=None, type=str,
                    help='model save path')
parser.add_argument('--batch_size', default=None, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--epochs', default=None, type=int,
                    help='train epoch number')

scaler = torch.cuda.amp.GradScaler()

def train(model,graph,loss_fn,optimizer,device,
          result_file_path,summary_writer,save_path,epochs=100,batch_size=32,train_loader=None,test_loader=None,
          scheduler=None):
    
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0

    for epoch in range(epochs):
        time_start = time.time ()
        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        steps = math.ceil(len(graph.train_mask) / batch_size)

        model.train()

        for batch_idx,(G1,G2,y,train_edge_id) in enumerate(train_loader):

            y_pred = model(G1.to(device),G2.to(device),graph,train_edge_id).squeeze()
            y = y.type(torch.FloatTensor).to(device)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(y_pred) > 0.5).type(torch.FloatTensor).to(device)
            metrics = Metrictor_PPI(y_pred.cpu().data,pre_result.cpu().data, y.cpu().data)
            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()


            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1
            print_file("epoch: {}, step: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                        .format(epoch,batch_idx, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))
    
        if global_step/5 ==0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                        os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_result_list = []
        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for batch_idx,(G1,G2,y,test_edge_id) in enumerate(test_loader):
                y_pred = model(G1.to(device),G2.to(device),graph,test_edge_id).squeeze()
                y = y.type(torch.FloatTensor).to(device)
                loss = loss_fn (y_pred,y)
                valid_loss_sum += loss.item()
                m = nn.Sigmoid ()
                pre_result = (m(y_pred) > 0.5).type(torch.FloatTensor).to(device)

                valid_result_list.append(y_pred.cpu().data)
                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(y.cpu().data)
                
                
        valid_loss = valid_loss_sum / valid_steps


        valid_result_list = torch.cat(valid_result_list,dim = 0)
        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)


        metrics = Metrictor_PPI(valid_result_list,valid_pre_result_list, valid_label_list)


        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps


        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']), save_file_path=result_file_path)
        
        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch


            torch.save({'epoch': epoch, 
                        'state_dict': model.state_dict()},
                        os.path.join(save_path, 'gnn_model_valid_best.ckpt'))
        
        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)

        print_file("epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {}, Best valid_f1: {}, in {} epoch"
                    .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1, global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)
        time_end = time.time ()
        time_sum = time_end - time_start
        print_file(str(time_sum),save_file_path=result_file_path)


args = parser.parse_args()
ppi_data = GNN_DATA(ppi_path=args.ppi_path)

print("----------------------- start split train and valid index -------------------")
print("whether to split new train and valid index file, {}".format(args.split_new))
if args.split_new:
    print("use {} method to split".format(args.split_mode))
ppi_data.split_dataset(args.train_valid_index_path, random_new=args.split_new, mode=args.split_mode)
print("----------------------- Done split train and valid index -------------------")

print("----------------------- get_feature_vec -----------------------")

ppi_data.get_prt_emb(pseq_path = args.pseq_path,emb_file = args.emb_path)


print("----------------------- get_ppi_net -----------------------")
ppi_data.generate_bigG_data ()
graph = ppi_data.data
print(graph.x.shape)

print("----------------------- get_contact_map and batch -----------------------")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
graph.to(device)


ppi_dict_r = ppi_data.ppi_dict_r
ppi_split_dict = ppi_data.ppi_split_dict
max_len = ppi_data.max_len
print(max_len)


train_dataset = MyDataset(ppi_dict_r,ppi_split_dict['train_index'],dmaproot=args.cp_path,device = device,max_len = max_len)
train_loader = DataLoader(dataset = train_dataset,batch_size = args.batch_size,shuffle = True,pin_memory=True,num_workers = 4)
test_dataset = MyDataset(ppi_dict_r,ppi_split_dict['valid_index'],dmaproot=args.cp_path,device = device,max_len = max_len)
test_loader = DataLoader(dataset = test_dataset,batch_size = args.batch_size,shuffle = False,pin_memory=True,num_workers = 4)


graph.train_mask = ppi_data.ppi_split_dict['train_index']
graph.val_mask = ppi_data.ppi_split_dict['valid_index']

print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))


model = CAN().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

scheduler = None
if args.use_lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

loss_fn = nn.BCEWithLogitsLoss().to(device)


if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

time_stamp = time.strftime("%Y-%m-%d %H_%M_%S")
save_path = os.path.join(args.save_path, "gnn_{}_{}".format(args.description, time_stamp))
result_file_path = os.path.join(save_path, "valid_results.txt")
config_path = os.path.join(save_path, "config.txt")
os.mkdir(save_path)

with open(config_path, 'w') as f:
    args_dict = args.__dict__
    for key in args_dict:
        f.write("{} = {}".format(key, args_dict[key]))
        f.write('\n')
    f.write('\n')
    f.write("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

summary_writer = SummaryWriter(save_path)

train(model, graph, loss_fn, optimizer, device,
    result_file_path, summary_writer, save_path,
    batch_size = args.batch_size, epochs=args.epochs,scheduler=scheduler,train_loader=train_loader,test_loader=test_loader
    )

summary_writer.close()


