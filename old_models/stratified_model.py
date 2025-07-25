#!/usr/env/bin python3

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
from sklearn.model_selection import  StratifiedKFold
import math
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=30)
parser.add_argument('--epochs', type=int, help='Epochs to train', default=20)
parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('-i', type=str, help='Input path', default='./tfrec_data_residues')
parser.add_argument('-c', type=str, help='Cluster information file path (.tsv format)', default='./cluster30.tsv')
parser.add_argument('-o', help='Output folder', type=str, default='./stratified')

def create_model(args, input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(input_shape, batch_size=args.batch_size, name='input'),
        tf.keras.layers.Dense(4096),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2048),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()
    return model

def build_model(args, model : tf.keras.Model, data_length):
    schedule = tf.keras.optimizers.schedules.CosineDecay(0.001, warmup_steps=100, decay_steps=(data_length // args.batch_size) * args.epochs)
    optim = tf.keras.optimizers.AdamW(learning_rate=schedule)
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.F1Score()
    ]

    model.compile(optimizer=optim,
                   loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                   metrics=metrics)
    
    return model

def decode_fn(record_bytes : bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,
      # Schema
      {
          "uniprot_id" : tf.io.FixedLenFeature((1,), dtype=tf.string),
          "embeddings": tf.io.FixedLenFeature((1024,), dtype=tf.float32),
          "sites": tf.io.FixedLenFeature((1,), dtype=tf.int64), }
  )


def load_data(path : str, tfrec=True):
    if tfrec:
        # In this case, path is a list of filenames
        tfrec_dataset = tf.data.TFRecordDataset(path).map(decode_fn)
        return tfrec_dataset

    return np.load(path)

def get_length(path : str):
    with open(f'{path}/n_elements.txt', 'r') as f:
        length = eval(f.read())
    return length

def split_train_test_clusters(args, clusters : pd.DataFrame, test_size : float):
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

def train_model(args, model : tf.keras.Model, data : tf.data.Dataset, clusters : pd.DataFrame, data_length : int):
    test_size = 0.2
    train_clusters, test_clusters = split_train_test_clusters(args, clusters, test_size)
    print('Clusters split')
    train_prots, test_prots = get_train_test_prots(clusters, train_clusters, test_clusters)
    print('Proteins split according to clusters')
    print(f'N train proteins: {len(train_prots)}')
    print(f'N test proteins: {len(test_prots)}')

    # Prepare test dataset
    test = data.filter(lambda x: x['uniprot_id'].ref() in test_prots).map(example_prep_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    print('Test data prepared')

    # Prepare train dataset
    train = data.filter(lambda x: x['uniprot_id'].ref() in train_prots).map(example_prep_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train = train.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    print('Train data prepared')

    print('Starting training')
    model.fit(train,  epochs=args.epochs, use_multiprocessing=True, workers=-1, 
                validation_data=test)
    loss, acc, f1 = model.evaluate(test, workers=-1, use_multiprocessing=True)
    print(f'Training finished with acc: {acc}, f1: {f1}')

    return model
def save_model(args, model : tf.keras.Model):
    now = datetime.now()

    name = now.strftime("stratified_%Y-%M-%d_%H:%M:%S")
    model.save(os.path.join(args.o, name), save_format='h5')

def example_prep_fn(example):
    return example['embeddings'], tf.one_hot(example['sites'][0], depth=2)

def prep_data_tfrec(data : tf.data.Dataset):
    return data.interleave(example_prep_fn, cycle_length=tf.data.AUTOTUNE).shuffle(buffer_size=1000, seed=42)

def load_clusters(path):
    return pd.read_csv(path, sep='\t', names=['cluster_rep', 'cluster_mem'])

def main(args):
    # Set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Load data from .npy array
    paths = glob.glob(f'{args.i}/*.tfrec')
    data = load_data(paths)
    print(args.c)
    clusters = load_clusters(args.c)
    data_length = get_length(args.i)
    # Prepare targets and split the data into inputs/outputs

    # Create the baseline model
    model = create_model(args, data.element_spec['embeddings'].shape)
    
    # Compile the model with an optimizer and a learning schedule
    build_model(args, model, data_length=data_length)

    # Train the model using 5-fold CV
    model = train_model(args, model, data, clusters, data_length)

    # Save the model as .h5
    save_model(args, model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
