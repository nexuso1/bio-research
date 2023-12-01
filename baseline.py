#!/usr/env/bin python3

import tensorflow as tf
import numpy as np
import argparse
import os
from datetime import datetime
from sklearn.model_selection import  StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=30)
parser.add_argument('--epochs', type=int, help='Epochs to train', default=20)
parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('-i', type=str, help='Input path', default='./small_dataset.npy')
parser.add_argument('-o', help='Output folder', type=str, default='./baseline')

def create_model(args, data):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((data.shape[1]), batch_size=args.batch_size, name='input'),
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
      {"embeddings": tf.io.FixedLenFeature([], dtype=tf.float32),
       "sites": tf.io.VarLenFeature(dtype=tf.int64) }
  )


def load_data(path, tfrec=True):
    if tfrec:
        trec_dataset = tf.data.TFRecordDataset([path]).map(decode_fn)
        
    return np.load(path)

def train_model(args, model : tf.keras.Model, X, y):
    kfold = StratifiedKFold(random_state=args.seed, shuffle=True)
    count = 0
    accs, f1s = [], []
    for train, test in kfold.split(X, np.argmax(y, axis=-1)):
        print(f'Starting training for fold {count}')
        model.fit(tf.gather(X, train), tf.gather(y, train), 
                  batch_size=args.batch_size, epochs=args.epochs, use_multiprocessing=True, workers=-1, 
                  validation_data=(tf.gather(X, test), tf.gather(y, test)))
        loss, acc, f1 = model.evaluate(tf.gather(X, test), tf.gather(y, test), batch_size=args.batch_size, workers=-1, use_multiprocessing=True)
        accs.append(acc)
        f1s.append(f1)

    print(f'Average accuracy: {np.sum(accs) / len(accs)}')
    print(f'Average F1: {np.sum(f1s) / len(f1s)}')

    return model
def save_model(args, model : tf.keras.Model):
    now = datetime.now()

    name = now.strftime("baseline_%Y-%M-%d_%H:%M:%S")
    model.save(os.path.join(args.o, name), save_format='h5')

def prep_data(data : np.ndarray):
    target = data[:, -1]
    target = tf.one_hot(target, depth=2)
    return tf.constant(data[:, :-1], dtype=tf.float32), target

def main(args):
    # Set random seed
    tf.random.set_seed(args.seed)

    # Load data from .npy array
    data = load_data(args.i)

    # Prepare targets and split the data into inputs/outputs
    X, y = prep_data(data)

    # Create the baseline model
    model = create_model(args, X)
    
    # Compile the model with an optimizer and a learning schedule
    build_model(args, model, data_length=X.shape[0])

    # Train the model using 5-fold CV
    model = train_model(args, model, X, y)

    # Save the model as .h5
    save_model(args, model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)