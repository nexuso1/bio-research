import pandas as pd
import numpy as np
import torch
import random
import numpy as np
import argparse
import json
import os

import torch.utils
import torch.utils.data
import torchmetrics
import datetime
import lora

from tqdm.auto import tqdm
from utils import remove_long_sequences, load_prot_data, Metadata, load_torch_model
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import set_seed, EsmModel, AutoTokenizer
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from classifiers import RNNTokenClassifer, RNNTokenClassiferConfig
from prot_dataset import ProteinTorchDataset
from functools import partial

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size)', default=8)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=20)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=1024)
parser.add_argument('--dataset_path', type=str, 
                     help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.',
                     default='data/phosphosite_sequences/phosphosite_df_small.json')
parser.add_argument('--clusters', type=str, help='Path to clusters', default='data/cluster30.tsv')
parser.add_argument('--fine_tune', action='store_true', help='Use fine tuning on the base model or not. Default is False', default=False)
parser.add_argument('--ft_only', action='store_true', help='Skip pre-training, only fine-tune', default=False)
parser.add_argument('--weight_decay', type=float, help='Weight decay', default=0.004)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=1)
parser.add_argument('--rnn', action='store_true', help='Use an RNN classification head', default=False)
parser.add_argument('--val_batch', type=int, help='Validation batch size', default=10)
parser.add_argument('--hidden_size', type=int, help='Classifier hidden size', default=256)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
parser.add_argument('-o', type=str, help='Output folder', default=None)
parser.add_argument('-n', type=str, help='Model name', default='esm')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model')
parser.add_argument('--lora', action='store_true', help='Use LoRA', default=False)
parser.add_argument('--dropout', type=float, help='Dropout probability', default=0)
parser.add_argument('--ft_epochs', type=int, help='Number of epochs for finetuning', default=10)
parser.add_argument('--type', help='ESM Model type', type=str, default='650M')
parser.add_argument('--pos_weight', help='Positive class weight', type=float, default=0.97)
parser.add_argument('--num_workers', help='Number of multiprocessign workers', type=int, default=0)
parser.add_argument('--rnn_layers', help='Number of RNN classifier layers', type=int, default=2)
parser.add_argument('--checkpoint_path', help='Resume training from checkpoint', type=str, default=None)
parser.add_argument('--model_path', help='Load model from this path (not a checkpoint)', type=str, default=None)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_esm(args):
    if args.type == '3B':
        model, tokenizer = EsmModel.from_pretrained('facebook/esm2_t36_3B_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
    elif args.type == '15B':
        model, tokenizer = EsmModel.from_pretrained('facebook/esm2_t48_15B_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')
    else:
        model, tokenizer = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    return model, tokenizer

def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

def load_clusters(path):
    return pd.read_csv(path, sep='\t', names=['cluster_rep', 'cluster_mem'])

def split_train_test_clusters(args, clusters : pd.DataFrame, test_size : float):
    reps = clusters['cluster_rep'].unique() # Unique cluster representatives
    train, test = train_test_split(reps, test_size=test_size, random_state=args.seed)
    return set(train), set(test)

def get_train_test_prots(clusters, train_clusters, test_clusters):
    train_mask = [x in train_clusters for x in clusters['cluster_rep']]
    test_mask = [x in test_clusters for x in clusters['cluster_rep']]
    train_prots = clusters['cluster_mem'][train_mask]
    test_prots = clusters['cluster_mem'][test_mask]
    return set(train_prots), set(test_prots)

def eval_model(model, test_ds, epoch, metrics : torchmetrics.MetricCollection):
    model.eval()
    metrics.reset()
    loss_metric =  torchmetrics.MeanMetric().to(device)
    epoch_message = f"Epoch={epoch+1}"
    progress_bar = tqdm(range(len(test_ds)))
    with torch.no_grad():
        for batch in test_ds:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model.predict(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], batch_lens=batch['batch_lens'], labels=batch['labels'])
            mask = batch['labels'].view(-1) != -100
            preds = torch.sigmoid(logits.view(-1)[mask])
            target = batch['labels'].view(-1)[mask]
            logs = compute_metrics(preds.view(-1, 1), target, metrics)
            loss_metric.update(loss)
            logs['loss'] = loss_metric.compute()
            message = [epoch_message] + [
                f"dev_{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}"
                for k, v in logs.items()
            ]
            progress_bar.set_description(" ".join(message))
            progress_bar.update(1)
    return logs

def train_model(args, train_ds : Dataset, dev_ds : Dataset, model : TokenClassifier, lr, metadata : Metadata=None, seed=42,
                start_epoch=0, optim=None):

    # Set all random seeds
    set_seeds(seed)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay) if optim is None else optim

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, len(train_ds) * args.epochs)

    metrics = {
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=-100).to(device),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=-100).to(device),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=-100).to(device),
    }
    loss_metric =  torchmetrics.MeanMetric().to(device)
    metrics = torchmetrics.MetricCollection(metrics)

    history = []
    # Train model
    for epoch in range(start_epoch, args.epochs):
        model.train()
        metrics.reset()
        epoch_message = f"Epoch={epoch+1}/{args.epochs}"
        # Progress bar
        data_and_progress = tqdm(
            train_ds,
            epoch_message,
            unit="batch",
            leave=False,
        )

        for i, batch in enumerate(train_ds):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model.train_predict(**batch)

            # Normalize the loss by the number of accumulation steps
            loss = loss / args.accum

            if args.accum == 1 or ( i > 0 and i % args.accum == 0) or (i + 1 == len(train_ds)):
                loss.backward()
                optim.step()
                schedule.step(epoch)
                optim.zero_grad()

                # Metrics logging
                logs = compute_metrics(logits, batch['labels'], metrics)
                loss_metric.update(loss)
                logs['loss'] = loss_metric.compute()
                message = [epoch_message] + [
                    f"{k}={v :.{0<abs(v)<2e-4 and '3g' or '4f'}}"
                    for k, v in logs.items()
                ]
                data_and_progress.set_description(" ".join(message))
            data_and_progress.update(1)

        save_checkpoint(args, model, config=model.config, optim=optim, epoch=epoch, loss=loss,
                         path=os.path.join(args.logdir, 'chkpt.pt'), metadata=metadata)
        print(f'Epoch {epoch}, starting evaluation...')
        eval_logs = eval_model(model, dev_ds, epoch, metrics)
        history.append(eval_logs)
        metadata.data['history'] = history
    return history, model

def preprocess_data(df : pd.DataFrame):
    """
    Preprocessing for Pbert/ProtT5. Replaces rare residues with 'X' and adds spaces between residues
    """
    df['sequence'] = df['sequence'].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    df['sequence'] = df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    return df

def split_dataset(data : pd.DataFrame, train_clusters, test_clusters):
    """
    Splits data into train and test data according to train and test clusters.
    """
    train_mask = data['id'].apply(lambda x: x in train_clusters)
    test_mask = data['id'].apply(lambda x: x in test_clusters)
    return data[train_mask], data[test_mask]

def save_as_string(obj, path):
    """
    Saves the given object as a JSON string.
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, 'w') as f:
        json.dump(obj, f)

def save_checkpoint(args, model : TokenClassifier, config : TokenClassifierConfig, optim : torch.optim.Optimizer,
                    epoch : int, loss, path : str, metadata  : Metadata = None):
    """
    Saves model checkpoint during training. Path should include the filename.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
    'optimizer_state_dict': optim.state_dict(),
    'model_state_dict': model.state_dict(),
    'args' : args,
    'config' : config,
    'epoch' : epoch,
    'loss' : loss 
    }, path)

    if metadata is not None:
        metadata.save(os.path.dirname(path))

def load_from_checkpoint(path):
    """
    Loads the model checkpoint from the given path. Creates a TokenClassifier instance, 
    with and passes it args from the checkpoint, and flags from the from the arguments.
    The created model loads the state dict from the checkpoint. Also creates an optimizer with 
    a state_dict from the checkpoint. 
    
    Returns a quintuple (model, opitm, epoch, loss, args)
    """
    chkpt = torch.load(path)
    args, epoch, loss = chkpt['args'], chkpt['epoch'], chkpt['loss']
    print(f'Checkpoint args: {args}')
    print(f'Checkpoint epoch: {epoch}')
    base, tokenizer = get_esm(args)
    config = chkpt['config']
    print(f'Checkpoint config: {config}')
    model = TokenClassifier(config, base)
    model.load_state_dict(chkpt['model_state_dict'])
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay)
    optim.load_state_dict(chkpt['optimizer_state_dict'])
    return model, tokenizer, optim, epoch, loss, args

def save_model(args, model : TokenClassifier, name : str):
    """
    Saves the model to the folder args.o if given, otherwise to args.logdir, with the given name.
    """
    if args.o is None:
        folder = args.logdir
    else:
        folder = args.o

    save_path = f'{folder}/{name}.pt'
    if not os.path.exists(f'{folder}'):
        os.mkdir(f'{folder}')

    model.save(save_path)
    print(f'Model saved to {save_path}')

def prep_batch(data, tokenizer, ignore_label=-100):
    """
    Collate function for a dataloader. "data" is a list of inputs.

    Return a dictionary with keys [input_ids, labels, batch_lens]
    """
    sequences, labels = zip(*data)
    batch = tokenizer(sequences, padding='longest', return_tensors="pt")
    sequence_length = batch["input_ids"].shape[1]
    # Pad the labels correctly
    batch['labels'] = np.array([[ignore_label] + list(label) + [ignore_label] * (sequence_length - len(label) - 1) for label in labels])
    batch['labels'] = torch.as_tensor(batch['labels'], dtype=torch.float32)
    batch['batch_lens'] = torch.as_tensor(np.array([len(x) for x in labels]))

    return batch

def compute_metrics(y_pred, y, metrics : torchmetrics.MetricCollection):
        """Compute and return metrics given the inputs, predictions, and target outputs."""
        metrics.update(y_pred, y.unsqueeze(-1))
        return metrics.compute()

def main(args):
    set_seeds(args.seed)

    # Create logdir name
    args.logdir = os.path.join(
        "logs",
        "{}_{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        ),
    )

    if args.checkpoint_path is not None:
        prev_ft_val = args.fine_tune
        model, tokenizer, optim, epoch, loss, args =  load_from_checkpoint(args.checkpoint_path)
     
    # Load a model saved with torch.save() 
    elif args.model_path is not None:
        model = load_torch_model(args.model_path)
    else:
        # Load ESM-2
        base, tokenizer = get_esm(args)

        # Create a classifier
        config = RNNTokenClassiferConfig(1, loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight])),
                                            hidden_size=args.hidden_size,
                                            n_layers=args.rnn_layers)
        if args.lora:
            config.apply_lora = args.lora, 
            config.lora_config=lora.MultiPurposeLoRAConfig(256)
        
        model = RNNTokenClassifer(config, base)
        if not args.lora:
            model.set_base_requires_grad(False)
    
    if args.compile:
        # Compile the model, useful in general on Ampere architectures and further
        compiled_model = torch.compile(model)
        compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
        training_model = compiled_model
    else:
        training_model = model.to(device)
    
    # Create metadata
    meta = Metadata()
    meta.data = {'args' : args }
    meta.save(args.logdir)

    # Load and preprocess data from the dataset
    data = load_prot_data(args.dataset_path)
    data = remove_long_sequences(data, args.max_length)
    #prepped_data = preprocess_data(data)

    # Load clustering information about proteins,
    # and split the clusters into train and test sets
    clusters = load_clusters(args.clusters)
    train_clusters, test_clusters = split_train_test_clusters(args, clusters, test_size=0.2) # Split clusters into train and test sets
    train_prots, test_prots = get_train_test_prots(clusters, train_clusters, test_clusters) # Extract the train proteins and test proteins
    train_df, test_df = split_dataset(data, train_prots, test_prots) # Split data according to the protein ids

    print(f'Train dataset shape: {train_df.shape}')
    print(f'Test dataset shape: {test_df.shape}')
    
    # Save test proteins
    test_path = f'./{args.o if args.o else args.logdir}/{args.n}_test_data.json'
    save_as_string(list(test_prots), test_path)
    print(f'Test prots saved to {test_path}')

    train_dataset = ProteinTorchDataset(train_df)
    dev_dataset = ProteinTorchDataset(test_df)

    train = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=partial(prep_batch, tokenizer=tokenizer),
                       persistent_workers=True if args.num_workers > 0 else False, num_workers=args.num_workers)
    dev = DataLoader(dev_dataset, args.batch_size, shuffle=True, collate_fn=partial(prep_batch, tokenizer=tokenizer),
                      persistent_workers=True if args.num_workers > 0 else False, num_workers=args.num_workers)


    
    # --- Training ---
    if args.checkpoint_path is not None:
        # Will potentially fine-tune the model, based on the loaded requires_grad params
        history, model = train_model(args, train, dev, model, lr = args.lr, start_epoch=epoch, optim=optim, meta=meta)
        args.fine_tune = prev_ft_val
        meta.data['history'] = history
    elif not args.fine_tune or args.fine_tune and not args.ft_only:
        history, compiled_model = train_model(args, train_ds=train, dev_ds=dev, model=training_model, seed=args.seed, lr=args.lr,
                                               metadata=meta)

    # --- Fine-tuning ---
    if args.fine_tune:
        print(f'batch {args.batch_size} accum {args.accum} effective batch {args.accum * args.batch_size}')
        # Save model before fine-tuning
        save_model(args, model, f'{args.n}_pre_ft')
        meta.data['fine_tuning'] = True
        # Unfreeze base
        model.set_base_requires_grad(True)

        if args.lora:
            model.apply_lora()

        # Recompile model
        if args.compile:
            compiled_model = torch.compile(model)
            compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
            training_model = compiled_model
        else:
            training_model = model.to(device)

        # Train with a lower learning rate
        ft_history, compiled_model = train_model(args, train_ds=train, dev_ds=dev, model=training_model,
                       seed=args.seed, lr=args.lr / 10, metadata=meta)
        history.extend(ft_history)
        meta.data['history'] = history
    save_model(args, model, args.n)
    return history, model

if __name__ == '__main__':
    args = parser.parse_args()
    history, model = main(args)