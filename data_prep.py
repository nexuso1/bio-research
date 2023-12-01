import tensorflow as tf
import glob
import numpy as np
import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p', help='File containing the phosphosite dataset', default='.\phosphosite_sequences\Phosphorylation_site_dataset')
parser.add_argument('-e', help='Directory containing embedding .npy files', default='.\sequences')
parser.add_argument('-o', help='Output dir', default='.\sequences')

def float_feature(value):
    """Returns a float_list from a list of float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int32_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_example(embed, sites):
    feature = {
        "embeddings": float_feature(embed),
        "sites" : int32_feature(sites)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_tfrecord_fn(example):
    feature_description = {
        "embeddings" : tf.io.FixedLenFeature([], tf.float32),
        "sites" : tf.io.VarLenFeature(dtype=tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

def serialize_data(args, data, idx):
    with tf.io.TFRecordWriter(
        f'{args.e}/embeds_{idx}.tfrec', 
    ) as writer:
        for example in data:
            writer.write(example.SerializeToString())

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
    buff_max_size = 1
    idx = 0
    for prot in glob.glob(f'{args.e}/*.npy'):
        if len(buffer) == buff_max_size:
            serialize_data(args, buffer, idx)
            idx += 1
            buffer = []
            break
    
        fixed_path = re.sub('\uf07c', '|', prot)
        fixed_path = re.sub('\uf03a', ':', fixed_path)
        seq_id = re.findall(r'GN:.+\|.+\|.+\|(.+)\.npy', fixed_path)[0]
        emebddings = np.load(prot)
        
        if not seq_id in phospho_data_pos:
            continue
        
        sites = [eval(i) - 1 for i in phospho_data_pos[seq_id]]
        targets = np.zeros(shape=(emebddings.shape[0]), dtype=np.int32)
        targets[sites] = 1
        targets = targets.reshape(-1, 1)
        example = create_example(emebddings, sites)
        buffer.append(example)

def main(args):
    phospho_data = pd.read_csv(args.p, sep='\t', skiprows=3)
    phospho_data_pos = extract_pos_info(phospho_data)
    prep_data(args, phospho_data_pos)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)