# SE3NET-PPI

Codes and models for the paper "Learning spatial structures and correlations of
proteins improves unknown protein–protein
interaction prediction".

Note: This codes is adapted from: https://github.com/lvguofeng/GNN_PPI/
## Dependencies
* python 
* tqdm
* pytorch
* numpy
* pandas
* pyg
* tensorboardx

## Data processing
### Protein embeddings
The package we use for generating protein embedding as following:
- https://github.com/sacdallago/bio_embeddings

The codes are in bio_embedding.py

### SE(3)-invariant matrix map
The package we use for generating the SE(3)-invariant matrix map as following:
- https://github.com/mdtraj/mdtraj

The codes are in SE3.py

## Dataset Download:

STRING(we use Homo sapiens subset): 
- PPI: https://stringdb-static.org/download/protein.actions.v11.0/9606.protein.actions.v11.0.txt.gz 
- Protein sequence: https://stringdb-static.org/download/protein.sequences.v11.0/9606.protein.sequences.v11.0.fa.gz

AlphaFold DB(we use Homo sapiens subset)
- https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v3.tar

## Using SE3NET-PPI

### Training

Training codes in my_train.py, and the run script in my_run_train.py.
- For the first time, you need to set the parameter random_new=True to generate a new data set division json file. (Otherwise, an error will be reported, No such file or directory: "./xxxx/string.bfs.fold1.json")
```
"python -u my_train.py \
    --description={} \              # Description of the current training task
    --ppi_path={} \                 # ppi dataset
    --pseq_path={} \                # protein sequence
    --emb_path={} \                 # protein embedding
    --SE3_path={} \                 # SE(3)-invariant matrix map
    --split_new={} \                # whether to generate a new data partition, or use the previous
    --split_mode={} \               # data split mode
    --train_valid_index_path={} \   # Data partition json file path
    --use_lr_scheduler={} \         # whether to use training learning rate scheduler
    --save_path={} \                # save model, config and results dir path
    --batch_size={} \               # Batch size
    --epochs={} \                   # Train epochs
    ".format(description,ppi_path, pseq_path,emb_path,SE3_path,vec_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path,
            batch_size, epochs)
```



### Testing

Testing codes in my_test.py and my_test_bigger.py, and the run script in my_run_test.py and my_run_test_bigger.py.

my_test.py: It can test the overall performance, and can also make in-depth analysis to test the performance of different test data separately.

my_test_bigger.py: It can test the performance between the trainset-homologous testset and the unseen testset. 
Running script my_run_test_bigger.py as above.




