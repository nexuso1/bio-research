import tensorflow as tf
import glob
import numpy as np
import re
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('-p', help='File containing the phosphosite dataset', default='./phosphosite_sequences/Phosphorylation_site_dataset')
parser.add_argument('-e', help='Directory containing embedding .npy files', default='./sequences')
parser.add_argument('-o', help='Output dir', default='./sequences')
parser.add_argument('-m', help='Mode. Can be either "per_prot" or "per_residue". per_prot saves all embeddings for a protein as one record, per_residues saves individual residue embeddings as records.', default='per_residue')

def float_feature(value):
    """Returns a float_list from a list of float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int32_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_example(id, embed, sites):
    feature = {
        "uniprot_id" : bytes_feature(id.encode('utf-8')),
        "embeddings": float_feature(embed),
        "sites" : int32_feature(sites)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_examples_residue(id, embed, sites):
    res = []
    for i, e in enumerate(embed):
        feature = {
        "uniprot_id" : bytes_feature(id.encode('utf-8')),
        "embeddings": float_feature(e),
        "target" : int32_feature([int(sites[i])]),
        "position" : int32_feature([i])
        }
        res.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    return res

def parse_tfrecord_fn(example):
    feature_description = {
        "uniprot_id" : tf.io.FixedLenFeature([], tf.string),
        "embeddings" : tf.io.FixedLenFeature([], tf.float32),
        "sites" : tf.io.VarLenFeature(dtype=tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

def serialize_data(args, data, idx):
    path = os.path.join(args.o, f'embeds_{idx}.tfrec')
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


def prep_data(args, phospho_data_pos):
    buffer = []
    buff_max_size = 150000
    idx = 0
    for prot in glob.glob(f'{args.e}/*.npy'):
        if len(buffer) >= buff_max_size:
            serialize_data(args, buffer, idx)
            idx += 1
            buffer = []
    
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
        if args.m == "per_residue":
            examples = create_examples_residue(seq_id, emebddings, targets)
            buffer.extend(examples)
        else:  
            example = create_example(seq_id, emebddings, sites)
            buffer.append(example)

    if len(buffer) > 0:
        serialize_data(args, buffer, idx)

def main(args):
    phospho_data = pd.read_csv(args.p, sep='\t', skiprows=3)
    phospho_data_pos = extract_pos_info(phospho_data)
    prep_data(args, phospho_data_pos)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
