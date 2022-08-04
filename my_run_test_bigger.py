import os


def run_func(description, ppi_path, pseq_path, emb_path,cp_path,
            index_path, gnn_model,result_file_path,batch_size, bigger_ppi_path, bigger_pseq_path):
    os.system("python my_test_bigger.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --emb_path={} \
            --cp_path={} \
            --index_path={} \
            --gnn_model={} \
            --result_file_path={} \
            --batch_size={} \
            --bigger_ppi_path={} \
            --bigger_pseq_path={} \
            ".format(description, ppi_path, pseq_path, emb_path,cp_path,
            index_path, gnn_model,result_file_path,batch_size, bigger_ppi_path, bigger_pseq_path))

if __name__ == "__main__":
    description = "test"

    ppi_path = "./data/actions/string_3000_aciton.tsv"
    pseq_path = "./data/dictionary/string_3000_new.tsv"
    emb_path = "./data/embedding/string_3000_ProtTransT5XLU50Embedder_all.npz"
    cp_path = "./data/STRING_SE3_npy/{}.npy"

    index_path = "./train_valid_index_json/string_3000.bfs.fold1.json"
    gnn_model = "./save_model/string_3000_bfs/gnn_model_valid_best.ckpt"

    result_file_path = './{}_result.txt'.format(description)
    batch_size = 16

    bigger_ppi_path = "./data/actions/STRING_action.tsv"
    bigger_pseq_path = "./data/dictionary/STRING.tsv"


    run_func(description, ppi_path, pseq_path, emb_path,cp_path,
            index_path, gnn_model,result_file_path,batch_size, bigger_ppi_path, bigger_pseq_path)
