import tensorflow as tf
import glob
import numpy as np
import re
import pandas as pd
import argparse
import os

from data_prep import create_examples_residue, create_example, float_feature, bytes_feature, int32_feature
from utils import save_as_string

parser = argparse.ArgumentParser()

parser.add_argument('-p', help='File containing the phosphosite dataset', default='./phosphosite_sequences/Phosphorylation_site_dataset')
parser.add_argument('-e', help='Directory containing embedding .npy files', default='./sequences')
parser.add_argument('-o', help='Output dir', default='./split_tfrec_data')
parser.add_argument('-c', help='Cluster path', default='cluster30.tsv')
parser.add_argument('-m', help='Mode. Can be either "per_prot" or "per_residue". per_prot saves all embeddings for a protein as one record, per_residues saves individual residue embeddings as records.', default='per_residue')

def serialize_data(path, data, idx):
    path = os.path.join(path, f'embeds_{idx}.tfrec')
    print(f'Saved to {path}. No. records: {len(data)}')
    with tf.io.TFRecordWriter(path) as writer:
        for example in data:
            writer.write(example.SerializeToString())
    
    print(f'Data saved in {path}')

def extract_pos_info(dataset : pd.DataFrame):
    """
    Extracts phosphoryllation site indices from the dataset. 
    Locations expected in the column 'MOD_RSD'.
    
    Returns a dictionary in format {ACC_ID : [list of phosphoryllation site indices]}
    """
    dataset['position'] = dataset['MOD_RSD'].str.extract(r'[\w]([\d]+)-p')
    grouped = dataset.groupby(dataset['ACC_ID'])
    res = {}
    for id, group in grouped:
        res[id] = group['position'].to_list()
    
    return res

def split_train_test_clusters(clusters : pd.DataFrame, test_size : float):
    reps = clusters['cluster_rep'].unique() # Unique cluster representatives
    np.random.shuffle(reps) # in-place shuffle
    train_last_idx = int(reps.shape[0] * (1 - test_size))
    train = reps[:train_last_idx]
    test = reps[train_last_idx:]

    return set(train), set(test)

def get_train_test_prots(clusters, train_clusters, test_clusters):
    train_mask = [x in train_clusters for x in clusters['cluster_rep']]
    test_mask = [x in test_clusters for x in clusters['cluster_rep']]
    train_prots = clusters['cluster_mem'].where(train_mask)
    test_prots = clusters['cluster_mem'].where(test_mask)

    return set(train_prots), set(test_prots)

def prep_data(args, phospho_data_pos, clusters):
    # Buffers that hold examples to be serialized
    train_buffer = []
    test_buffer = []

    train_clusters, test_clusters = split_train_test_clusters(clusters, 0.2)
    train_prots, test_prots = get_train_test_prots(clusters, train_clusters, test_clusters)
    
    # Save train and test protein ids separately in a file
    save_as_string(list(train_prots), f'{args.o}/train_prots.json')
    save_as_string(list(test_prots), f'{args.o}/test_prots.json')

    # Counters for number of examples in each dataset
    n_test = 0
    n_train = 0

    buff_max_size = 150000
    idx = 0
    for prot in glob.glob(f'{args.e}/*.npy'):
        if len(train_buffer) >= buff_max_size:
            serialize_data(os.path.join(args.o, 'train'), train_buffer, idx)
            idx += 1
            n_train += len(train_buffer)
            train_buffer = []

        if len(test_buffer) >= buff_max_size:
            serialize_data(os.path.join(args.o, 'test'), test_buffer, idx)
            idx += 1
            n_test += len(test_buffer)
            test_buffer = []
    
        seq_id = prot.split('_')[-1][:-4] # Shave off .npy
        if not seq_id in phospho_data_pos:
            continue

        try:
            emebddings = np.load(prot)
        except:
            print(f"Could not load {prot}, skipping")
            continue

        sites = [eval(i) - 1 for i in phospho_data_pos[seq_id]]
        targets = np.zeros(shape=(emebddings.shape[0]), dtype=np.int32)
        try:
            targets[sites] = 1
        except:
            print(f"Bad mapping of sites in {prot}. Sites = {sites}, prot.shape = {emebddings.shape}")
        targets = targets.reshape(-1, 1)
        
        if seq_id in test_prots:
            buffer = test_buffer
        else:
            buffer = train_buffer

        if args.m == "per_residue":
            examples = create_examples_residue(seq_id, emebddings, targets)
            buffer.extend(examples)
        else:  
            example = create_example(seq_id, emebddings, sites)
            buffer.append(example)

    # Serialize the remainder in the buffers
    if len(train_buffer) > 0:
        serialize_data(os.path.join(args.o, 'train'), train_buffer, idx)

    if len(test_buffer) > 0:
        serialize_data(os.path.join(args.o, 'test'), test_buffer, idx)

    # Save number of elements in the dataset separately
    with open(f'{args.o}/train/n_elements.txt', 'w') as f:
        f.write(str(n_train))

    with open(f'{args.o}/test/n_elements.txt', 'w') as f:
        f.write(str(n_test))

def load_clusters(path):
    return pd.read_csv(path, sep='\t', names=['cluster_rep', 'cluster_mem'])

def main(args):
    phospho_data = pd.read_csv(args.p, sep='\t', skiprows=3)
    phospho_data_pos = extract_pos_info(phospho_data)
    clusters = load_clusters(args.c)
    prep_data(args, phospho_data_pos, clusters)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
