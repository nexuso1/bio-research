#!/usr/env/bin python3

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import os
import glob
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=30)
parser.add_argument('--epochs', type=int, help='Epochs to train', default=20)
parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('-i', type=str, help='Train data folder', default='./split_tfrec_data/train')
parser.add_argument('-t', type=str, help='Test data folder', default='/split_tfrec_data/test')
parser.add_argument('-n', type=str, help='Model name (without the file extension)', default='focal_tf_model')
parser.add_argument('-o', help='Output folder', type=str, default='/storage/praha1/home/nexuso1/models')
parser.add_argument('--lr', help='Initial learning rate', type=float, default=0.001)
parser.add_argument('--alpha', help='Fraction of initial learning rate to be used as a floor in the cosine decay', type=float, default=0.01)
parser.add_argument('--layers', type=str, help='Sequential classifier layers shape.', default='[2048,2048,1024]')

def create_model(args, input_shape):
    inputs = tf.keras.layers.Input(input_shape, name='input')
    last = inputs
    for layer in args.layers:
        if isinstance(layer, str):
            units = eval(layer)
        else:
            units = int(layer)

        last = tf.keras.layers.BatchNormalization()(last)
        last = tf.keras.layers.Dense(units, activation='relu')(last)
        last = tf.keras.layers.Dropout(0.2)(last)

    bn = tf.keras.layers.BatchNormalization()(last)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(bn)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def build_model(args, model : tf.keras.Model, data_length):
    schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, alpha=args.alpha, decay_steps=(data_length // args.batch_size) * args.epochs)
    optim = tf.keras.optimizers.AdamW(learning_rate=schedule)
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.F1Score(),
        tf.keras.metrics.Precision(),
        tf.keras.metris.Recall()
    ]

    model.compile(optimizer=optim,
                   loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=3.0, apply_class_balancing=False),
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
          "target": tf.io.FixedLenFeature((1,), dtype=tf.int64),
          "position": tf.io.FixedLenFeature((1,), dtype=tf.int64) }
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

def train_model(args, model : tf.keras.Model, train_data : tf.data.Dataset, test_data : tf.data.Dataset):
    callbacks = [tf.keras.callbacks.EarlyStopping(
    monitor='val_f1_score',
    min_delta=0,
    patience=10,
    verbose=0,
    mode='max',
    restore_best_weights=True,
    start_from_epoch=2
)]

    model.fit(train_data,  epochs=args.epochs, use_multiprocessing=True, workers=-1, 
                batch_size=args.batch_size, validation_data=test_data, callbacks=callbacks)
    loss, acc, f1 = model.evaluate(test_data, workers=-1, use_multiprocessing=True, batch_size=args.batch_size)
    print(f'Training finished with acc: {acc}, f1: {f1}')

    return model

def save_model(args, model : tf.keras.Model):
    model.save(os.path.join(args.o, f'{args.n}.h5'), save_format='h5')

def example_prep_fn(example):
    return example['embeddings'], example['target']

def load_clusters(path):
    return pd.read_csv(path, sep='\t', names=['cluster_rep', 'cluster_mem'])

def prepare_dataset(data : tf.data.Dataset):
    return data.map(example_prep_fn).batch(args.batch_size).prefetch(tf.data.AUTOTUNE).shuffle(1100, seed=42)

def main(args):
    # Set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Load tfrec data from the given folders
    paths = glob.glob(f'{args.i}/*.tfrec') # train paths
    train_data = load_data(paths)
    paths = glob.glob(f'{args.t}/*.tfrec') # test paths
    test_data = load_data(paths)

    # Load the data length information
    data_length = get_length(args.i)

    # Create the baseline model
    model = create_model(args, train_data.element_spec['embeddings'].shape)

    # Prepare data
    train_data = prepare_dataset(train_data)
    test_data = prepare_dataset(test_data)
    
    # Compile the model with an optimizer and a learning schedule
    build_model(args, model, data_length=data_length)

    # Train the model
    model = train_model(args, model, train_data, test_data)

    # Save the model as .h5
    save_model(args, model)

if __name__ == '__main__':
    args = parser.parse_args()
    args.layers = ast.literal_eval(args.layers)
    main(args)
