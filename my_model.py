import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random

from torch_geometric.nn import GINConv


class DMCNN (torch.nn.Module):

    def __init__(self,emb_dim,channels):
        super (DMCNN,self).__init__ ()
        self.embedding_size = emb_dim
        self.channels = channels
        self.conv1 = nn.Conv2d (in_channels = self.embedding_size,out_channels = self.channels,kernel_size = 3,bias = False)
        self.batchnorm = nn.BatchNorm2d (self.channels)
        self.global_pool_16 = nn.AdaptiveAvgPool2d (4)
        self.global_pool_4 = nn.AdaptiveAvgPool2d (2)
        self.global_pool_1 = nn.AdaptiveAvgPool2d (1)
        self.flatten = nn.Flatten ()
        self.lin = nn.Linear (21,256)
        self.norm = nn.BatchNorm1d(256)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.lin.reset_parameters()
        self.fc.reset_parameters()


    def forward(self,x):
        x = self.conv1 (x)
        x = self.batchnorm (x)
        g_16 =  self.flatten(self.global_pool_16(x))
        g_4 = self.flatten(self.global_pool_4(x))
        g_1 =  self.flatten(self.global_pool_1(x))
        x = torch.cat([g_16, g_4,g_1],1)
        x = self.lin(F.relu(x))
        x = self.norm(x)

        return x

class CAN(torch.nn.Module):
    def __init__(self, in_feature=1024,dm_emb=1,cnn_hidden=1,
                 hidden=512, train_eps=True,
                 class_num=7):
        super(CAN, self).__init__()
        self.train_eps = train_eps

        #seq
        self.lin0 = nn.Linear(in_feature, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.batchnorm1 = nn.BatchNorm1d(hidden)

        #SE3_cnn
        self.dmcnn = DMCNN(dm_emb,cnn_hidden)


        #PPI-NET
        self.gin_conv = GINConv(
            nn.Sequential(
                nn.Linear( hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3= nn.Linear(hidden, hidden//2)
        self.batchnorm2 = nn.BatchNorm1d(hidden//2)


        #out
        self.fc = nn.Linear(hidden*2, class_num)

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()


        self.gin_conv.reset_parameters()

        self.fc2.reset_parameters()

    def forward(self,g1,g2,G,train_edge_id, p=0.5):
        #pre_emb
        x = self.lin0(G.x)
        x = self.lin1(F.relu(x))
        x = self.batchnorm1(x)

        #cp
        g1 = self.dmcnn(g1)
        g2 = self.dmcnn(g2)

        #ppi-net
        x = self.gin_conv(x, G.edge_index)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin3(x)

        #fusion
        node_id = G.edge_index[:,train_edge_id]

        x1 = torch.cat([x[node_id[0]],g1],dim = 1)
        x2 = torch.cat([x[node_id[1]],g2],dim = 1)

        x = torch.cat([torch.abs(x1-x2),torch.mul(x1,x2)],dim=1)
        x = self.fc(x)

        return x.float()