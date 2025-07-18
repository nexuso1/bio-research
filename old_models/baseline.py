#!/usr/env/bin python3

import tensorflow as tf
import numpy as np
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
parser.add_argument('-i', type=str, help='Input path', default='./tfrec_data_residue')
parser.add_argument('-o', help='Output folder', type=str, default='./baseline')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def create_model(args, input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(input_shape, batch_size=args.batch_size, name='input'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
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

def decode_fn(record_bytes):
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

def load_data(path, tfrec=True):
    if tfrec:
        # In this case, path is a list of filenames
        tfrec_dataset = tf.data.TFRecordDataset(path).map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
        return tfrec_dataset

    return np.load(path)

def get_length(path):
    with open(f'{path}/n_elements.txt', 'r') as f:
        length = eval(f.read())
    return length

def train_model(args, model : tf.keras.Model, data, data_length : tf.data.Dataset):
    accs, f1s = [], []
    folds = 5
    test_size = math.floor(data_length/folds)
    for i in range(folds):
        print(f'Starting training for fold {i}')
        test = data.skip(test_size * i).take(test_size)
        test = test.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        train = data.take(i * test_size).concatenate(data.skip((i + 1) * test_size))
        train = train.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        model.fit(train,  epochs=args.epochs, use_multiprocessing=True, workers=-1, 
                  validation_data=test)
        loss, acc, f1 = model.evaluate(test, workers=-1, use_multiprocessing=True)
        accs.append(acc)
        f1s.append(f1)

    print(f'Average accuracy: {np.sum(accs) / len(accs)}')
    print(f'Average F1: {np.sum(f1s) / len(f1s)}')

    return model

def save_model(args, model : tf.keras.Model):
    now = datetime.now()

    name = now.strftime("baseline_%Y-%M-%d_%H_%M_%S")
    model.save(os.path.join(args.o, name), save_format='h5')

def example_prep_fn(example):
    return example['embeddings'], tf.one_hot(example['sites'][0], depth=2)

def prep_data_tfrec(data : tf.data.Dataset):
    return data.map(example_prep_fn, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=1000, seed=42)

def prep_data_numpy(data : np.ndarray):
    target = data[:, -1]
    target = tf.one_hot(target, depth=2)
    return tf.constant(data[:, :-1], dtype=tf.float32), target

def main(args):
    # Set random seed
    tf.random.set_seed(args.seed)

    # Load data from .npy array
    paths = glob.glob(f'{args.i}/*.tfrec')
    data = load_data(paths)
    data_length = get_length(args.i)
    # Prepare targets and split the data into inputs/outputs
    prepared_ds = prep_data_tfrec(data)

    # Create the baseline model
    model = create_model(args, data.element_spec['embeddings'].shape)
    
    # Compile the model with an optimizer and a learning schedule
    build_model(args, model, data_length=data_length)

    # Train the model using 5-fold CV
    model = train_model(args, model, prepared_ds, data_length)

    # Save the model as .h5
    save_model(args, model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
