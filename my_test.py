import os
import time
import math
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm

from my_data import *
from my_model import CAN
from my_utils import Metrictor_PPI, print_file


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Test Model')
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
parser.add_argument('--index_path', default=None, type=str,
                    help='cnn_rnn and gnn unified train and valid ppi index')
parser.add_argument('--gnn_model', default=None, type=str,
                    help="gnn trained model")
parser.add_argument('--batch_size', default=None, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--test_all', default='False', type=boolean_string,
                    help="test all or test separately")


def test(model, graph,test_loader,device):
    valid_pre_result_list = []
    valid_label_list = []


    model.eval()
    with torch.no_grad():
        for batch_idx, (G1, G2, y, test_edge_id) in enumerate(test_loader):
            y_pred = model(G1.to(device), G2.to(device), graph, test_edge_id).squeeze()
            y = y.type(torch.FloatTensor).to(device)

            m = nn.Sigmoid()
            pre_result = (m(y_pred) > 0.5).type(torch.FloatTensor).to(device)

            valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(y.cpu().data)


    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
    valid_label_list = torch.cat(valid_label_list, dim=0)

    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)

    metrics.show_result()

    print("Recall: {}, Precision: {}, F1: {}".format(metrics.Recall, metrics.Precision, metrics.F1))


def main():
    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)

    ppi_data.get_prt_emb(
        pseq_path=args.pseq_path, emb_file=args.emb_path)

    ppi_data.generate_bigG_data()

    ppi_dict_r = ppi_data.ppi_dict_r

    max_len = ppi_data.max_len

    num2protein_name = {v: k for k, v in ppi_data.protein_name.items()}

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2
    # fake_edge_num = len(ppi_data.fake_edge) // 2
    fake_edge_num = 0

    with open(args.index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']

    graph.val_mask = index_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    node_vision_dict = {}

    for index in graph.train_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 1
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 1

    for index in graph.val_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 0
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 0


    vision_num = 0
    unvision_num = 0
    for node in node_vision_dict:
        if node_vision_dict[node] == 1:
            vision_num += 1
        elif node_vision_dict[node] == 0:
            unvision_num += 1
    print("vision node num: {}, unvision node num: {}".format(vision_num, unvision_num))

    test1_mask = []
    test2_mask = []
    test3_mask = []

    for index in graph.val_mask:
        ppi = ppi_list[index]
        temp = node_vision_dict[ppi[0]] + node_vision_dict[ppi[1]]
        if temp == 2:
            test1_mask.append(index)
        elif temp == 1:
            test2_mask.append(index)
        elif temp == 0:
            test3_mask.append(index)
    print("test1 edge num: {}, test2 edge num: {}, test3 edge num: {}".format(len(test1_mask), len(test2_mask),
                                                                              len(test3_mask)))

    graph.test1_mask = test1_mask
    graph.test2_mask = test2_mask
    graph.test3_mask = test3_mask

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CAN().to(device)

    model.load_state_dict(torch.load(args.gnn_model)['state_dict'])

    graph.to(device)


    if args.test_all:
        print("---------------- valid-test-all result --------------------")
        test_dataset = MyDataset(ppi_dict_r, graph.val_mask, dmaproot=args.cp_path, device=device,
                                 max_len=max_len)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=4)
        test(model, graph,test_loader,device)
    else:
        print("---------------- valid-test1 result --------------------")
        if len(graph.test1_mask) > 0:
            test_dataset = MyDataset(ppi_dict_r, graph.test1_mask, dmaproot=args.cp_path, device=device,
                                     max_len=max_len)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                     num_workers=4)
            test(model, graph, test_loader, device)
        print("---------------- valid-test2 result --------------------")
        if len(graph.test2_mask) > 0:
            test_dataset = MyDataset(ppi_dict_r, graph.test2_mask, dmaproot=args.cp_path, device=device,
                                     max_len=max_len)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                     num_workers=4)
            test(model, graph, test_loader, device)
        print("---------------- valid-test3 result --------------------")
        if len(graph.test3_mask) > 0:
            test_dataset = MyDataset(ppi_dict_r, graph.test3_mask, dmaproot=args.cp_path, device=device,
                                     max_len=max_len)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                     num_workers=4)
            test(model, graph, test_loader, device)


if __name__ == "__main__":
    main()
