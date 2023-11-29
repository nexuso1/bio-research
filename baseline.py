#!/usr/env/bin python3

import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=30)
parser.add_argument('--epochs', type=int, help='Epochs to train', default=20)
parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('-i', type=str, help='Input path')
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
def load_data(path):
    return np.load(path)

def train_model(model):
    ...
def save_model():
    ...
def main(args):
    dataset = load_data(args.i)
    model = create_model(args, dataset)
    build_model(args, model, data_length=dataset.shape[0])
    model = train_model(model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)