import numpy as np
import mdtraj as md
import itertools
import pandas as pd

#获取蛋白名称
file = pd.read_csv(r'E:\Alphafold\STRING.tsv',sep='\t',header=None)
all_p_list = file[0].to_list()

for map_name in all_p_list:
    pdb = md.load_pdb('E:\Alphafold\pdb\{}.pdb'.format(map_name))
    carbon_alphas = pdb.topology.select("protein and name == 'CA'")
    all_pairs = [pair for pair in itertools.combinations(carbon_alphas, 2)]
    distances = md.compute_distances(pdb, atom_pairs=all_pairs)
    distance_map = np.zeros((len(carbon_alphas),len(carbon_alphas)))
    distance_map[np.triu_indices(len(carbon_alphas),1)] = distances
    distance_map = distance_map + distance_map.T
    np.save(r'E:\Alphafold\SE3_npy\{}.npy'.format(map_name), distance_map)