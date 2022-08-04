import os
import time
import argparse
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def run_func(description,ppi_path, pseq_path,emb_path,cp_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path,
            batch_size, epochs):
    os.system("python -u my_train.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --emb_path={} \
            --cp_path={} \
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --batch_size={} \
            --epochs={} \
            ".format(description,ppi_path, pseq_path,emb_path,cp_path,
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path,
                    batch_size, epochs))

if __name__ == "__main__":
    description = "string_3000_bfs_1"
    ppi_path = "./data/string_3000_aciton.tsv"

    pseq_path = "./data/string_3000_new.tsv"
    emb_path ="./data/embedding/STRING_ProtTransT5XLU50Embedder_all.npz"
    cp_path = "./data/STRING_SE3_npy/{}.npy"


    split_new = "False"
    split_mode = "bfs"
    train_valid_index_path = "./train_valid_index_json/string_3000.bfs.fold1.json"

    use_lr_scheduler = "True"
    save_path = "./String_3000/"
    batch_size = 256
    epochs = 50

    run_func(description,ppi_path, pseq_path, emb_path,cp_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler,save_path,batch_size,epochs)