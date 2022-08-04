import os


def run_func(description, ppi_path, pseq_path,emb_path,cp_path,
            index_path, gnn_model,batch_size,test_all):
    os.system("python my_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --emb_path={} \
            --cp_path={} \
            --index_path={} \
            --gnn_model={} \
            --batch_size={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path,emb_path,cp_path,
                    index_path, gnn_model, batch_size,test_all))

if __name__ == "__main__":
    description = "test"

    ppi_path = "./data/actions/string_3000_aciton.tsv"
    pseq_path = "./data/dictionary/string_3000_new.tsv"
    emb_path = "./data/embedding/string_3000_ProtTransT5XLU50Embedder_all.npz"
    cp_path = "./data/string_3000_SE3/{}.npy"


    index_path = "./train_valid_index_json/string_3000.bfs.fold1.json"
    gnn_model = "./save_model/string_3000_bfs/gnn_model_valid_best.ckpt"
    batch_size = 256
    test_all = "False"

    # test test

    run_func(description, ppi_path, pseq_path,emb_path,cp_path,index_path, gnn_model, batch_size,test_all)
