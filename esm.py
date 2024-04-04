import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import json
import os
import lora
import re

from tqdm.auto import tqdm
from utils import remove_long_sequences, load_prot_data
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import set_seed, EsmModel, AutoTokenizer
from torcheval.metrics import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Maximum batch size (in number of residues)', default=2048)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=20)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=1024)
parser.add_argument('--dataset_path', type=str, 
                     help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.',
                     default='./phosphosite_sequences/phosphosite_df_small.json')
parser.add_argument('--clusters', type=str, help='Path to clusters', default='cluster30.tsv')
parser.add_argument('--fine_tune', action='store_true', help='Use fine tuning on the base model or not. Default is False', default=False)
parser.add_argument('--weight_decay', type=float, help='Weight decay', default=0.004)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=1)
parser.add_argument('--rnn', type=bool, help='Use an RNN classification head', default=False)
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
parser.add_argument('--ft_epochs', type=int, help='Number of epochs for finetuning', default=10)
parser.add_argument('--type', help='ESM Model type', type=str, default='650M')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

class TokenClassifier(nn.Module):
    """
    Model that consist of a base embedding model, and a token classification head at the end, using 
    the last hidden states as its input.
    """
    ignore_index = -100 # Ignore labels with index -100
    token_model = None

    def __init__(self, args, base_model : nn.Module, dropout = 0.2, n_labels = 1, fine_tune=False, use_lora=False) -> None:
        super(TokenClassifier, self).__init__()
        self.base = base_model

        if fine_tune and use_lora:
            self.apply_lora()
        
        self.n_labels = n_labels
        # Focal loss for each element that will be summed
        # self.loss = partial(focal_loss.sigmoid_focal_loss, reduction='sum')
        # BCE Loss with weight 95 for the positive class. In the dataset, the 
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.95]))
        self.dropout = nn.Dropout(dropout)
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
        
        lstm = nn.LSTM(self.base.config.hidden_size, hidden_size=args.hidden_size, bidirectional=True, batch_first=True)
        outputs = nn.Linear(args.hidden_size, self.n_labels)
        self.classifier = nn.Sequential(
            lstm,
            outputs
        )

    def build_cnn_classifier(self, args):
        layer_conf = [(self.base.config.hidden_size, 128, 1, 1),
                      (128, 192, 5, 2),
                      (192, 256, 5, 2),
                      (256, 512, 3, 1),
                      ]
        
        layers = []
        for in_channels, out_channels, k, stride in layer_conf:
            layers.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=stride))
            layers.append(torch.nn.BatchNorm1d(out_channels))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.AdaptiveAvgPool1d(1))

        self.sequence_rep_head = torch.nn.Linear(self.base.config.hidden_size, args.hidden_size)
        self.token_model = torch.nn.Sequential(*layers)

        self.classifier = torch.nn.Sequential([
            torch.nn.Linear(layer_conf[-1][1] + args.hidden_size, 1)
        ])
        
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        batch_lens=None,
        output_hidden_states=None,
        return_dict = False,
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
        # Generate per-sequence representations via averaging
#        NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_reps = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_reps.append(sequence_output[i, 1 : tokens_len + 1].mean(0))

        sequence_reps = torch.vstack(sequence_reps)
        sequence_reps = self.sequence_rep_head(sequence_reps)
        sequence_output = self.dropout(sequence_output)
        if self.token_model:
            token_features = self.token_model(sequence_output)
        else:
            token_features = sequence_output

        classifier_features = []
        for i in range(batch_lens.shape[0]):
            # Broadcast the sequence features to match the 0-th dimension of the corresponding sequence
            broadcasted_seq_features = sequence_reps[i].unsqueeze(0).broadcast_to(sequence_output[i].shape[0], -1)

            # Concatenate the features together, resulting in a tensor of shape 
            # (sequence_length, base_hidden_size + sequence_feature_dim)
            catted = torch.cat((token_features[i], broadcasted_seq_features), -1)

            classifier_features.append(catted)
        
        # Combine the features for every sequence in the batch
        classifier_features = torch.stack(classifier_features)
        logits = self.classifier(classifier_features)
        loss = None

        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.ignore_index).type_as(labels)
                )
                valid_logits=active_logits[active_labels!=-100].flatten()
                valid_labels=active_labels[active_labels!=-100]
                loss = self.loss(valid_logits, valid_labels)
            else:
                loss = self.loss(logits.view(-1, self.n_labels), labels.view(-1))

        if not return_dict:
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

def eval_model(model, test_ds, epoch):
    metrics = [
        (BinaryF1Score(threshold=0.5, device=device), 'f1'),
        (BinaryPrecision(threshold=0.5, device=device), 'precision'),
        (BinaryRecall(threshold=0.5, device=device), 'recall'),
        (BinaryConfusionMatrix(threshold=0.5, device=device), 'confusion matrix')
    ]

    model.eval()
    progress_bar = tqdm(range(len(test_ds)))
    with torch.no_grad():
        for batch in test_ds:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Model returns a tuple, logits are the first element when not given labels
            preds = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], batch_lens=batch['batch_lens'])
            mask = batch['labels'].view(-1) != -100
            preds = torch.sigmoid(preds[0].view(-1)[mask])
            target = batch['labels'].view(-1)[mask]
            for metric, _ in metrics:
                metric.update(target=target.int(), input=preds)

            progress_bar.update(1)
    print(f'Epoch {epoch}:')
    
    res = {}
    for metric, name in metrics:
        val = metric.compute().detach().cpu().numpy()
        print(f'    {name}: {val}')
        res[name] = val

def train_model(args, train_ds : Dataset, test_ds : Dataset, model : torch.nn.Module, tokenizer,
                lr, epochs, batch, val_batch, accum, seed=42, deepspeed=None):

    # Set all random seeds
    set_seeds(seed)

    optim = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CyclicLR(optim, gamma=0.95, max_lr=lr, base_lr=lr*0.01, mode='exp_range')
    #schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, len(train_ds) * epochs)
    progress_bar = tqdm(range(len(train_ds) * epochs))

    # Train model
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_ds):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(training=True, **batch)
            if accum == 1 or ( i > 0 and i % accum == 0):
                outputs[0].backward()
                optim.step()
                schedule.step(epoch)
                optim.zero_grad()
            progress_bar.update(1)

        print(f'Epoch {epoch}, starting evaluation...')
        eval_model(model, test_ds, epoch)
    return tokenizer, model 

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
    
    train_dataset = create_dataset(tokenizer=tokenizer, seqs=list(train_df['sequence']), labels=list(train_df['label']),
                                max_batch_residues=args.batch_size) # Create a huggingface dataset
    test_dataset = create_dataset(tokenizer=tokenizer, seqs=list(test_df['sequence']), labels=list(test_df['label']), 
                                max_batch_residues=args.batch_size)

    model = TokenClassifier(args, base, use_lora=False, fine_tune=False)
    if args.compile:
        compiled_model = torch.compile(model)
        compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
        training_model = compiled_model
    else:
        training_model = model.to(device)
    tokenizer, compiled_model = train_model(args, train_ds=train_dataset, test_ds=test_dataset, model=training_model, tokenizer=tokenizer,
                       seed=args.seed, batch=args.batch_size, val_batch=args.val_batch, epochs=args.epochs, accum=args.accum, lr=args.lr)

    if args.fine_tune:
        save_model(args, model, f'{args.o}_pre_ft.pt')
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
        tokenizer, compiled_model = train_model(args, train_ds=train_dataset, test_ds=test_dataset, model=training_model, tokenizer=tokenizer,
                       seed=args.seed, batch=args.batch_size, val_batch=args.val_batch, epochs=args.ft_epochs, accum=args.accum, lr=args.lr / 10)
        
    return tokenizer, model

if __name__ == '__main__':
    args = parser.parse_args()
    tokenizer, model = main(args)
    save_model(args, model, args.n)
