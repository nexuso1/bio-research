import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import json
import os

import torch.utils
import torch.utils.data
import lora
import re
import torchmetrics

from tqdm.auto import tqdm
from utils import remove_long_sequences, load_prot_data
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import set_seed, EsmModel, AutoTokenizer
from modules import Unet1D, RNNClassifier
from prot_dataset import ProteinTorchDataset
from functools import partial

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size)', default=4)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=20)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=1024)
parser.add_argument('--dataset_path', type=str, 
                     help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.',
                     default='./phosphosite_sequences/phosphosite_df_small.json')
parser.add_argument('--clusters', type=str, help='Path to clusters', default='cluster30.tsv')
parser.add_argument('--fine_tune', action='store_true', help='Use fine tuning on the base model or not. Default is False', default=False)
parser.add_argument('--weight_decay', type=float, help='Weight decay', default=0.004)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=1)
parser.add_argument('--rnn', type=bool, help='Use an RNN classification head', default=True)
parser.add_argument('--val_batch', type=int, help='Validation batch size', default=10)
parser.add_argument('--hidden_size', type=int, help='Classifier hidden size. Relevant for cnn, rnn and simple classifiers', default=256)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
parser.add_argument('-o', type=str, help='Output folder', default='output')
parser.add_argument('-n', type=str, help='Model name', default='esm.pt')
parser.add_argument('--layers', type=str, help='Hidden layers for the linear classifier', default='[1024]')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model')
parser.add_argument('--lora', action='store_true', help='Use LoRA', default=False)
parser.add_argument('--cnn', action='store_true', help='Use CNN classifier', default=False)
parser.add_argument('--mlp', action='store_true', help='Use an MLP classifier', default=False)
parser.add_argument('--dropout', type=float, help='Dropout probability', default=0)
parser.add_argument('--ft_epochs', type=int, help='Number of epochs for finetuning', default=10)
parser.add_argument('--type', help='ESM Model type', type=str, default='650M')
parser.add_argument('--pos_weight', help='Positive class weight', type=float, default=0.98)
parser.add_argument('--num_workers', help='Number of multiprocessign workers', type=int, default=0)
parser.add_argument('--rnn_layers', help='Number of RNN classifier layers', type=int, default=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

class TokenClassifier(nn.Module):
    """
    Model that consist of a base embedding model, and a token classification head at the end, using 
    the last hidden states as its input.
    """
    ignore_index = -100 # Ignore labels with index -100
    token_model = None

    def __init__(self, args, base_model : nn.Module, n_labels = 1, fine_tune=False, use_lora=False) -> None:
        super(TokenClassifier, self).__init__()
        # Embedding model
        self.base = base_model

        if fine_tune and use_lora:
            self.apply_lora()
        
        self.n_labels = n_labels
        # Use sequence representations as an input to the classifier
        self.use_seq_reps = True 
        # Focal loss for each element that will be summed
        # self.loss = partial(focal_loss.sigmoid_focal_loss, reduction='sum')
        # BCE Loss with weight 95 for the positive class. In the dataset, the 

        # Use positive class weights
        pos_weight = None
        if args.pos_weight:
            pos_weight = torch.Tensor([args.pos_weight])
        
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dropout = nn.Dropout(args.dropout)
        self.classifier_requires_lens = False
        if args.rnn:
            self.build_rnn_classifier(args)
        elif args.cnn:
            self.build_cnn_classifier(args)
        elif args.mlp:
            self.build_linear_classifier(args)
        else:
            self.build_simple_classifier(args)

        if not fine_tune:
            # Freeze base model parameters
            self.freeze_base()

    def apply_lora(self, config=lora.MultiPurposeLoRAConfig(rank=256)):
        self.lora_config = config
        self.base = lora.modify_with_lora(self.base, self.lora_config)

        # Freeze base model parameters, except LoRA
        for (param_name, param) in self.base.named_parameters():
            param.requires_grad = False       

        for (param_name, param) in self.base.named_parameters():
                if re.fullmatch(self.lora_config.trainable_param_names, param_name):
                    param.requires_grad = True

        print('LoRA applied.')

    def init_weights(self, m):
        """
        Uses xavier/glorot weight initialization for linear layers, and 0 for bias
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def unfreeze_base(self):
        """
        Unfreeze the base model (i.e. for finetuning purposes)
        """
        for p in self.base.parameters():
          p.requires_grad = True

    def build_cnn_classifier(self, args):
            configs = [
                Unet1D.LayerConfig(self.base.config.hidden_size, 256, 7, 2, 2),
                Unet1D.LayerConfig(256, 384, 5, 2, 2),
                Unet1D.LayerConfig(384, 512, 3, 2, 2),
                Unet1D.LayerConfig(512, 1024, 3, 2, 2)
            ]
            self.classifier = Unet1D(configs, 1)
            self.use_seq_reps = False

    def build_simple_classifier(self, args):
        self.sequence_rep_head = torch.nn.Sequential(
            torch.nn.Linear(self.base.config.hidden_size, args.hidden_size),
            torch.nn.BatchNorm1d(args.hidden_size),
            torch.nn.ReLU(inplace=True)
        )

        self.classifier = torch.nn.Linear(self.base.config.hidden_size + args.hidden_size, self.n_labels)
        self.init_weights(self.sequence_rep_head)
        self.init_weights(self.classifier)

    def build_rnn_classifier(self, args):
        self.classifier = RNNClassifier(self.base.config.hidden_size * 2, self.n_labels, args.hidden_size, args.rnn_layers)
        self.classifier_requires_lens = True
        self.init_weights(self.classifier)

    def build_linear_classifier(self, args):
        self.classifier = nn.Sequential(
            nn.Linear(self.base.config.hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_labels)
        )

    def freeze_base(self):
        """
        Freeze the base model
        """
        for p in self.base.parameters():
          p.requires_grad = False

    def get_sequence_reps(self, sequence_output : torch.Tensor, batch_lens):
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        pad_mask = torch.arange(0, sequence_output.shape[0], device=device)[:, None, None].expand_as(sequence_output)
        lens_reshaped = batch_lens[:, None, None].expand_as(pad_mask)
        pad_mask = pad_mask > lens_reshaped # True if a given position is padding
        # Zero the padding values
        sequence_output[pad_mask] = 0
        # Calculate the sequence means
        seq_rep = torch.mean(sequence_output, 1)
        # Add the 'sequence' dim
        seq_rep = seq_rep.unsqueeze(1) # (B, 1, CH)
        # Repeat the means for every sequence element (i.e. sequence length-times),
        seq_rep = seq_rep.expand_as(sequence_output) # (B, S, CH)

        return torch.cat([sequence_output, seq_rep], -1) # (B, S, 2CH)

        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        batch_lens=None,
        training = False, 
        **kwargs
    ):
        if training:
            self.base.train()
        outputs = self.base(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Last hidden state
        sequence_output = outputs[0] 
        sequence_output = self.dropout(sequence_output)

        classifier_features = sequence_output
        if self.use_seq_reps:
            classifier_features = self.get_sequence_reps(classifier_features, batch_lens)
        
        if self.classifier_requires_lens:
            logits = self.classifier(classifier_features, torch.sum(attention_mask, -1))
        else:
            logits = self.classifier(classifier_features)
        loss = None

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.reshape(-1, self.n_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.ignore_index).type_as(labels)
                )
                valid_logits=active_logits[active_labels!=-100].flatten()
                valid_labels=active_labels[active_labels!=-100]
                loss = self.loss(valid_logits, valid_labels)
            else:
                loss = self.loss(logits.view(-1, self.n_labels), labels.view(-1))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output

        #return {
            #'loss' : loss,
            #'logits' : logits,
           # 'hidden_states' : outputs.hidden_states,
          #  'attentions' : outputs.attentions,
         #   'outputs' : (loss, outputs)
        #}

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

def batch_data(seqs, labels, tokenizer, max_batch_length=2048):
    batch_elements = 0
    max_in_batch = 0
    batches = []
    buffer = []
    buf_labels = []

    ignore_label = -100
    for i in range(len(seqs)):
        element_length = len(labels[i])
        if element_length > max_in_batch:
            new_batch_length = element_length * (batch_elements + 1)
        else:
            new_batch_length = max_in_batch * (batch_elements + 1)
        
        # Check if adding the element exceeds the limit (assuming we pad to the longest element)
        if new_batch_length >= max_batch_length:
            batch = tokenizer(buffer, padding='longest', return_tensors="pt")
            sequence_length = batch["input_ids"].shape[1]
            batch['labels'] = np.array([[ignore_label] + list(label) + [ignore_label] * (sequence_length - len(label) - 1) for label in buf_labels])
            batch['labels'] = torch.as_tensor(batch['labels'], dtype=torch.float32)
            batch['batch_lens'] = torch.as_tensor(np.array([len(x) for x in buf_labels]))
            batches.append(batch)
            buffer = []
            buf_labels = []
            batch_elements = 0
            max_in_batch = 0

        buffer.append(seqs[i])
        buf_labels.append(labels[i])
        
        if max_in_batch < element_length:
            max_in_batch = element_length

        batch_elements += 1

    if len(buffer) != 0:
        batch = tokenizer(buffer, padding='longest', return_tensors="pt")
        sequence_length = batch["input_ids"].shape[1]
        batch['labels'] = np.array([[ignore_label] + list(label) + [ignore_label] * (sequence_length - len(label) - 1) for label in buf_labels])
        batch['labels'] = torch.as_tensor(batch['labels'], dtype=torch.float32)
        batch['batch_lens'] = torch.as_tensor(np.array([len(x) for x in buf_labels]))
        batches.append(batch)

    return batches

def create_dataset(tokenizer, seqs, labels, max_batch_residues):
    batch_seqs = batch_data(seqs, labels, tokenizer, max_batch_residues)
    # tokenized = tokenizer(batch_seqs, padding='longest')
    #dataset = Dataset.from_buffer(tokenized)
    # we need to cut of labels after max_length positions for the data collator to add the correct padding ((max_length - 1) + 1 special tokens)
    #dataset = dataset.add_column("labels", batched_labels)
    #dataset = batch_data(dataset.shuffle(seed=42), max_batch_residues)
    return batch_seqs

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
            # Model returns a tuple, logits are the first element when not given labels
            loss, logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], batch_lens=batch['batch_lens'])
            mask = batch['labels'].view(-1) != -100
            preds = torch.sigmoid(preds[0].view(-1)[mask])
            target = batch['labels'].view(-1)[mask]
            for metric, _ in metrics:
                metric.update(target=target.int(), input=preds)
            logs = compute_metrics(logits, batch['labels'], metrics)
            loss_metric.update(loss)
            logs['loss'] = loss_metric.compute()
            message = [epoch_message] + [
                f"dev_{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}"
                for k, v in logs.items()
            ]
            progress_bar.set_description(" ".join(message))

    return logs

def train_model(args, train_ds : Dataset, test_ds : Dataset, model : torch.nn.Module, tokenizer,
                lr, epochs, batch, val_batch, accum, seed=42, deepspeed=None):

    # Set all random seeds
    set_seeds(seed)

    optim = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
    # Cycle momentum not available with adam
    schedule = torch.optim.lr_scheduler.CyclicLR(optim, gamma=0.99, max_lr=lr, base_lr=lr*0.01,
                                                  mode='exp_range',cycle_momentum=False)
    #schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, len(train_ds) * epochs)


    metrics = {
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=-100).to(device),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=-100).to(device),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=-100).to(device)
    }
    loss_metric =  torchmetrics.MeanMetric().to(device)
    metrics = torchmetrics.MetricCollection(metrics)

    history = []
    # Train model
    for epoch in range(epochs):
        model.train()
        metrics.reset()
        epoch_message = f"Epoch={epoch+1}/{epochs}"
        # Progress bar
        data_and_progress = tqdm(
            train_ds,
            epoch_message,
            unit="batch",
            leave=False,
        )
        for i, batch in enumerate(train_ds):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model(training=True, **batch)
            if accum == 1 or ( i > 0 and i % accum == 0):
                loss.backward()
                optim.step()
                schedule.step(epoch)
                optim.zero_grad()
                # Metrics logging
                logs = compute_metrics(logits, batch['labels'], metrics)
                loss_metric.update(loss)
                logs['loss'] = loss_metric.compute()
                message = [epoch_message] + [
                    f"{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}"
                    for k, v in logs.items()
                ]
            data_and_progress.set_description(" ".join(message))

        print(f'Epoch {epoch}, starting evaluation...')
        metrics = eval_model(model, test_ds, epoch, metrics)
        history.append(metrics)
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

def save_model(args, model, name):
    save_path = f'{args.o}/{name}.pt'
    if not os.path.exists(f'{args.o}'):
        os.mkdir(f'{args.o}')

    torch.save(model, save_path)
    print(f'Model saved to {save_path}')

def prep_batch(data, tokenizer, ignore_label=-100):
    """
    Collate function for a dataloader. "data" is a list of inputs
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
    base, tokenizer = get_esm(args)
    data = load_prot_data(args.dataset_path)
    data = remove_long_sequences(data, args.max_length)
    #prepped_data = preprocess_data(data)
    clusters = load_clusters(args.clusters)
    train_clusters, test_clusters = split_train_test_clusters(args, clusters, test_size=0.2) # Split clusters into train and test sets
    train_prots, test_prots = get_train_test_prots(clusters, train_clusters, test_clusters) # Extract the train proteins and test proteins
    train_df, test_df = split_dataset(data, train_prots, test_prots) # Split data according to the protein ids
    print(f'Train dataset shape: {train_df.shape}')
    print(f'Test dataset shape: {test_df.shape}')
    
    test_path = f'./{args.o}/{args.n}_test_data.json'
    save_as_string(list(test_prots), test_path)
    print(f'Test prots saved to {test_path}')
    
    # train_dataset = create_dataset(tokenizer=tokenizer, seqs=list(train_df['sequence']), labels=list(train_df['label']),
    #                             max_batch_residues=args.batch_size) # Create a huggingface dataset
    # test_dataset = create_dataset(tokenizer=tokenizer, seqs=list(test_df['sequence']), labels=list(test_df['label']), 
    #                             max_batch_residues=args.batch_size)

    train_dataset = ProteinTorchDataset(train_df)
    test_dataset = ProteinTorchDataset(test_df)

    train = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=partial(prep_batch, tokenizer=tokenizer),
                       persistent_workers=True if args.num_workers > 0 else False, num_workers=args.num_workers)
    test = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=partial(prep_batch, tokenizer=tokenizer),
                      persistent_workers=True if args.num_workers > 0 else False, num_workers=args.num_workers)


    model = TokenClassifier(args, base, use_lora=False, fine_tune=False)
    if args.compile:
        compiled_model = torch.compile(model)
        compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
        training_model = compiled_model
    else:
        training_model = model.to(device)
    history, compiled_model = train_model(args, train_ds=train, test_ds=test, model=training_model, tokenizer=tokenizer,
                       seed=args.seed, batch=args.batch_size, val_batch=args.val_batch, epochs=args.epochs, accum=args.accum, lr=args.lr)

    if args.fine_tune:
        save_model(args, model, f'{args.n}_pre_ft')
        # Unfreeze base
        model.unfreeze_base()

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
        ft_history, compiled_model = train_model(args, train_ds=train_dataset, test_ds=test_dataset, model=training_model, tokenizer=tokenizer,
                       seed=args.seed, batch=args.batch_size, val_batch=args.val_batch, epochs=args.ft_epochs, accum=args.accum, lr=args.lr / 10)

    save_model(args, model, args.n)
    return history, model

if __name__ == '__main__':
    args = parser.parse_args()
    history, model = main(args)

