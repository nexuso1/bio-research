import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise
import keras
import torch
import argparse
import train_script as ts

from train_script import TokenClassifier

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=10)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=1024)
parser.add_argument('--fasta', type=str, help='Path to the FASTA protein database', default='./phosphosite_sequences/Phosphosite_seq.fasta')
parser.add_argument('--phospho', type=str, help='Path to the phoshporylarion dataset', default='./phosphosite_sequences/Phosphorylation_site_dataset')
parser.add_argument('--dataset_path', type=str, help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.', default='./phosphosite_sequences/phosphosite_df.json')
parser.add_argument('--pretokenized', type=bool, help='Input dataset is already pretokenized', default=False)
parser.add_argument('--val_batch', type=int, help='Validation batch size', default=2)
parser.add_argument('--clusters', type=str, help='Path to clusters', default='clusters_30.csv')
parser.add_argument('--fine_tune', type=bool, help='Use fine tuning on the base model or not. Default is False', default=False)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=4)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
parser.add_argument('-o', type=str, help='Output folder', default='output')
parser.add_argument('-n', type=str, help='Model name', default='prot_model.pt')

def create_model(args, bert, tokenizer):
    inputs = keras.Input(shape=(None), batch_size=args.batch_size)

    tokenizer_layer = keras.layers.TorchModuleWrapper(tokenizer, name='tokenizer')
    tokenizer_layer = tokenizer(inputs)
    ids = tokenizer_layer['input_ids']
    type_ids = tokenizer_layer['type_ids']
    attention_mask = tokenizer_layer['attention_mask']
    bert_layer = keras.layers.TorchModuleWrapper(bert)([ids, type_ids, attention_mask])

    model = keras.Model(inputs, bert_layer)
    return model

def build_model(args, model : keras.Model, data_length):
    schedule = keras.optimizers.schedules.CosineDecay(0.001, warmup_steps=100, decay_steps=(data_length // args.batch_size) * args.epochs)
    optim = keras.optimizers.AdamW(learning_rate=schedule)
    metrics = [
        keras.metrics.CategoricalAccuracy(),
        keras.metrics.F1Score()
    ]

    model.compile(optimizer=optim,
                   loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                   metrics=metrics)
    
    return model

def prepare_data(args):
    data = ts.load_data(args.dataset_path)
    data = ts.remove_long_sequences(data, args.max_length)
    prepped_data = ts.preprocess_data(data)
    clusters = ts.load_clusters(args.clusters)
    train_clusters, test_clusters = ts.split_train_test_clusters(args, clusters, test_size=0.2) # Split clusters into train and test sets
    train_prots, test_prots = ts.get_train_test_prots(clusters, train_clusters, test_clusters) # Extract the train proteins and test proteins
    train_df, test_df = ts.split_dataset(prepped_data, train_prots, test_prots) # Split data according to the protein ids
    print(f'Train dataset shape: {train_df.shape}')
    print(f'Test dataset shape: {test_df.shape}')
    
    test_path = f'./{args.o}/{args.n}_test_data.json'
    ts.save_as_string(list(test_prots), test_path)
    print(f'Test prots saved to {test_path}')

    return train_df, test_df

def main(args):
    # train_df, test_df = prepare_data(args)
    pbert, tokenizer = ts.get_bert_model()
    model = create_model(args, pbert, tokenizer)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)