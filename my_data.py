import os
import json
import numpy as np
import pandas as pd
import copy
import torch
import random
from tqdm import tqdm
from my_utils import *
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch


class GNN_DATA:
    def __init__(self, ppi_path, exclude_protein_path=None, max_len=1200, skip_head=True, p1_index=0, p2_index=1,
                 label_index=2, graph_undirection=True, bigger_ppi_path=None):
        self.ppi_list = []
        self.ppi_dict = {}
        self.ppi_dict_r = {}
        self.ppi_label_list = []
        self.protein_dict = {}
        self.protein_name = {}
        self.ppi_path = ppi_path
        self.bigger_ppi_path = bigger_ppi_path
        self.max_len = max_len

        name = 0
        ppi_name = 0
        # maxlen = 0
        self.node_num = 0
        self.edge_num = 0
        if exclude_protein_path != None:
            with open(exclude_protein_path, 'r') as f:
                ex_protein = json.load(f)
                f.close()
            ex_protein = {p: i for i, p in enumerate(ex_protein)}
        else:
            ex_protein = {}

        for line in tqdm(open(ppi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split('\t')

            if line[p1_index] in ex_protein.keys() or line[p2_index] in ex_protein.keys():
                continue

            # get node and node name
            if line[p1_index] not in self.protein_name.keys():
                self.protein_name[line[p1_index]] = name
                name += 1

            if line[p2_index] not in self.protein_name.keys():
                self.protein_name[line[p2_index]] = name
                name += 1

            # get edge and its label
            class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5,
                         'expression': 6}
            temp_data = ""
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]

            if temp_data not in self.ppi_dict.keys():
                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0, 0, 0, 0, 0, 0, 0]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list.append(temp_label)
                self.ppi_dict_r[ppi_name] = temp_data + "__" + str(temp_label)
                ppi_name += 1
            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[label_index]]] = 1
                self.ppi_label_list[index] = temp_label
                self.ppi_dict_r[index] = temp_data + "__" + str(temp_label)

            if bigger_ppi_path != None:
                skip_head = True
                for line in tqdm(open(bigger_ppi_path)):
                    if skip_head:
                        skip_head = False
                        continue
                    line = line.strip().split('\t')

                    if line[p1_index] not in self.protein_name.keys():
                        self.protein_name[line[p1_index]] = name
                        name += 1

                    if line[p2_index] not in self.protein_name.keys():
                        self.protein_name[line[p2_index]] = name
                        name += 1

                    temp_data = ""
                    if line[p1_index] < line[p2_index]:
                        temp_data = line[p1_index] + "__" + line[p2_index]
                    else:
                        temp_data = line[p2_index] + "__" + line[p1_index]

                    if temp_data not in self.ppi_dict.keys():
                        self.ppi_dict[temp_data] = ppi_name
                        temp_label = [0, 0, 0, 0, 0, 0, 0]
                        temp_label[class_map[line[label_index]]] = 1
                        self.ppi_label_list.append(temp_label)
                        self.ppi_dict_r[ppi_name] = temp_data + "__" + str(temp_label)
                        ppi_name += 1
                    else:
                        index = self.ppi_dict[temp_data]
                        temp_label = self.ppi_label_list[index]
                        temp_label[class_map[line[label_index]]] = 1
                        self.ppi_label_list[index] = temp_label
                        self.ppi_dict_r[index] = temp_data + "__" + str(temp_label)

        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):
            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)

        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]
            # print(len(self.protein_name))
            self.ppi_list[i][0] = self.protein_name[seq1_name]
            self.ppi_list[i][1] = self.protein_name[seq2_name]

        if graph_undirection:
            for i in tqdm(range(ppi_num)):
                temp_ppi = self.ppi_list[i][::-1]
                temp_ppi_label = self.ppi_label_list[i]
                # if temp_ppi not in self.ppi_list:
                self.ppi_list.append(temp_ppi)
                self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=False, mode='random'):
        if random_new:
            if mode == 'random':
                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1 - test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1 - test_size)):]

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':
                print("use {} methed split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(train_valid_index_path, 'r') as f:
                self.ppi_split_dict = json.load(f)
                f.close()

    def get_protein_aac(self, pseq_path):
        # aac: amino acid sequences

        self.pseq_path = pseq_path
        self.pseq_dict = {}
        self.protein_len = []

        for line in tqdm(open(self.pseq_path)):
            line = line.strip().split('\t')
            if line[0] not in self.pseq_dict.keys():
                self.pseq_dict[line[0]] = line[1]
                self.protein_len.append(len(line[1]))

        print("protein num: {}".format(len(self.pseq_dict)))
        print("protein average length: {}".format(np.average(self.protein_len)))
        print("protein max & min length: {}, {}".format(np.max(self.protein_len), np.min(self.protein_len)))


    def get_prt_emb(self, pseq_path, emb_file):
        self.get_protein_aac(pseq_path)
        data = np.load(emb_file)
        self.pvec_dict = {}
        self.dim = 1024

        for p_name in tqdm(self.pseq_dict.keys()):
            self.pvec_dict[p_name] = data[p_name]

        for name in tqdm(self.protein_name.keys()):
            self.protein_dict[name] = self.pvec_dict[name]

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ppi_ndary = np.array(self.ppi_list)
        for edge in ppi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)

    def generate_bigG_data(self):
        self.get_connected_num()

        print("Connected domain num: {}".format(self.ufs.count))

        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long)
        self.x = []
        i = 0
        for name in self.protein_name:
            assert self.protein_name[name] == i
            i += 1
            self.x.append(self.protein_dict[name])

        self.x = np.array(self.x)
        self.x = torch.tensor(self.x, dtype=torch.float)
        self.data = Data(x=self.x, edge_index=self.edge_index.T, edge_attr_1=self.edge_attr)


def default_loader(dpath, pid, max_len, device):
    dmap = np.load(dpath.format(pid))
    dmap_len = dmap.shape[0]
    if dmap_len >= max_len:
        dmap = dmap[:max_len, :max_len]
    else:
        pad_len = (max_len - dmap_len) // 2
        if dmap_len % 2 != 0:
            dmap = np.pad(
                dmap, ((pad_len + 1, pad_len), (pad_len + 1, pad_len)), 'constant', constant_values=(0, 0)
            )
        else:
            dmap = np.pad(dmap, ((pad_len, pad_len), (pad_len, pad_len)), 'constant', constant_values=(0, 0))

    assert dmap.shape == (max_len, max_len)
    G = torch.from_numpy(dmap).type(torch.FloatTensor).unsqueeze(0)
    return G


class MyDataset(Dataset):

    def __init__(self, ppi_dict_r,mask, dmaproot, device, max_len, loader=default_loader):
        pns = []
        super(MyDataset, self).__init__()
        for i in mask:
            p1, p2, labal = ppi_dict_r[i].split('__')
            train_id = i
            labal = [int(i) for i in list(labal) if i.isdigit()]
            pns.append((p1, p2, labal, int(train_id)))
        self.pns = pns
        self.loader = loader
        self.dmaproot = dmaproot
        self.max_len = max_len
        self.device = device

    def __getitem__(self, index):
        p1, p2, label, train_id = self.pns[index]
        G1 = self.loader(self.dmaproot, p1, self.max_len, self.device)
        G2 = self.loader(self.dmaproot, p2, self.max_len, self.device)
        return G1, G2, torch.tensor(label).type(torch.FloatTensor), torch.tensor(train_id)

    def __len__(self):
        return len(self.pns)





