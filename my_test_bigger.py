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
parser.add_argument('--result_file_path', default=None, type=str,
                    help="result_file_path")
parser.add_argument('--batch_size', default=None, type=int,
                    help="gnn train batch size, edge batch size")
parser.add_argument('--bigger_ppi_path', default=None, type=str,
                    help="if use bigger ppi")
parser.add_argument('--bigger_pseq_path', default=None, type=str,
                    help="if use bigger ppi")


def test(model, graph, test_loader,result_file_path,device):
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

    print_file("Recall: {}, Precision: {}, F1: {}".format(metrics.Recall, metrics.Precision, metrics.F1),
               result_file_path)


def main():
    args = parser.parse_args()


    ppi_data = GNN_DATA(ppi_path=args.ppi_path, bigger_ppi_path=args.bigger_ppi_path)

    ppi_data.get_prt_emb(
        pseq_path=args.pseq_path, emb_file=args.emb_path)

    ppi_data.generate_bigG_data()

    ppi_dict_r = ppi_data.ppi_dict_r

    max_len = ppi_data.max_len

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()

    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2


    with open(args.index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']

    all_mask = [i for i in range(truth_edge_num)]
    graph.val_mask = list(set(all_mask).difference(set(graph.train_mask)))

    print_file("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)),
               args.result_file_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CAN().to(device)

    model.load_state_dict(torch.load(args.gnn_model)['state_dict'])

    graph.to(device)

    test_dataset = MyDataset(ppi_dict_r, graph.val_mask, dmaproot=args.cp_path, device=device,
                             max_len=max_len)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=0)
    test(model, graph, test_loader, args.result_file_path, device)


if __name__ == "__main__":
    main()
