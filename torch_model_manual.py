import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import evaluate
import json
import os
import torcheval

from tqdm.auto import tqdm
from datetime import datetime
from utils import remove_long_sequences, load_prot_data
from datasets import Dataset, IterableDataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, set_seed
from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Maximum batch size (in number of residues)', default=2048)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=50)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=1024)
parser.add_argument('--fasta', type=str, help='Path to the FASTA protein database', default='./phosphosite_sequences/Phosphosite_seq.fasta')
parser.add_argument('--phospho', type=str, help='Path to the phoshporylarion dataset', default='./phosphosite_sequences/Phosphorylation_site_dataset')
parser.add_argument('--dataset_path', type=str, help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.', default='./phosphosite_sequences/phosphosite_df_small.json')
parser.add_argument('--pretokenized', type=bool, help='Input dataset is already pretokenized', default=False)
parser.add_argument('--clusters', type=str, help='Path to clusters', default='cluster30.tsv')
parser.add_argument('--fine_tune', type=bool, help='Use fine tuning on the base model or not. Default is False', default=False)
parser.add_argument('--weight_decay', type=float, help='Weight decay', default=0.004)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=1)
parser.add_argument('--rnn', type=bool, help='Use an RNN classification head', default=False)
parser.add_argument('--hidden_size', type=int, help='RNN hidden size. Only relevant when --rnn=True.', default=256)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
parser.add_argument('-o', type=str, help='Output folder', default='output')
parser.add_argument('-n', type=str, help='Model name', default='prot_model.pt')

# os.environ['TORCH_COMPILE_DEBUG'] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch._inductor.config.compile_threads = 16
print(device)

class TokenClassifier(nn.Module):
    """
    Model that consist of a base embedding model, and a token classification head at the end, using 
    the last hidden state as its output.
    """
    def __init__(self, base_model : nn.Module, dropout = 0.2, n_labels = 2, fine_tune=False) -> None:
        super(TokenClassifier, self).__init__()
        self.base = base_model
        self.n_labels = n_labels

        if args.rnn:
            self.build_rnn_classifier(args)
        else:
            self.build_linear_classifier(args)

        if not fine_tune:
            self.freeze_base()

    def build_rnn_classifier(self, args):
        lstm = nn.LSTM(self.base.config.hidden_size, hidden_size=args.hidden_size, bidirectional=True, batch_first=True)
        outputs = nn.Linear(args.hidden_size, self.n_labels)(lstm)
        self.classifier = outputs

    def build_linear_classifier(self, args):
        self.classifier = nn.Sequential(
            nn.Linear(self.base.config.hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_labels)
        )

    def freeze_base(self):
        for p in self.base.parameters():
          p.requires_grad = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict = False,
    ):
        outputs = self.base(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=0.1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))

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

def get_bert_model():
    pbert = BertModel.from_pretrained("Rostlab/prot_bert")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    return pbert, tokenizer

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

    pad_token_id = -100
    for i in range(len(seqs)):
        element_length = len(labels[i])
        if element_length > max_in_batch:
            new_batch_length = element_length * (batch_elements + 1)
        else:
            new_batch_length = max_in_batch * (batch_elements + 1)
        
        # Check if adding the element exceeds the limit (assuming we pad to the longest element)
        if new_batch_length > max_batch_length:
            batch = tokenizer(buffer, padding='longest', return_tensors="pt")
            sequence_length = batch["input_ids"].shape[1]
            batch['labels'] = [[pad_token_id] + list(label) + [pad_token_id] * (sequence_length - len(label) - 1) for label in buf_labels]
            batch['labels'] = torch.tensor(batch['labels'], dtype=torch.int64)
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
        batch['labels'] = [[pad_token_id] + list(label) + [pad_token_id] * (sequence_length - len(label) - 1) for label in buf_labels]
        batch['labels'] = torch.tensor(batch['labels'], dtype=torch.int64)
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
    f1 = MulticlassF1Score(device=device, average='macro')
    with torch.no_grad():
        for batch in test_ds:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(input_ids=batch['input_ids'],
                           attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
            mask = batch['labels'].view(-1) != -100
            preds = torch.argmax(preds[0], -1).view(-1)
            f1 = f1.update(target=batch['labels'].view(-1)[mask], input=preds[mask])

    print(f'Epoch {epoch}, F1: {f1.compute().detach().cpu().numpy()}')

def train_model(args, train_ds, test_ds, model : torch.nn.Module, tokenizer,
                lr, epochs, batch, val_batch, accum, seed=42, deepspeed=None):

    # Set all random seeds
    set_seeds(seed)

    optim = torch.optim.AdamW(model.parameters(), weight_decay=0.004)
    schedule = torch.optim.lr_scheduler.CyclicLR(optim, gamma=0.99, max_lr=lr, base_lr=lr*0.01, mode='exp_range',cycle_momentum=False)
    progress_bar = tqdm(range(len(train_ds) * epochs))
    # Train model
    for epoch in range(epochs):
        for i, batch in enumerate(train_ds):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs[0].backward()
            optim.step()
            schedule.step(epoch)
            optim.zero_grad()
            progress_bar.update(1)

        eval_model(model, test_ds, epoch)
    return tokenizer, model

def preprocess_data(df : pd.DataFrame):
    """
    Preprocessing for Pbert/ProtT5
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


def main(args):
    pbert, tokenizer = get_bert_model()
    if not args.pretokenized:
        data = load_prot_data(args.dataset_path)
        data = remove_long_sequences(data, args.max_length)
        prepped_data = preprocess_data(data)
        clusters = load_clusters(args.clusters)
        train_clusters, test_clusters = split_train_test_clusters(args, clusters, test_size=0.2) # Split clusters into train and test sets
        train_prots, test_prots = get_train_test_prots(clusters, train_clusters, test_clusters) # Extract the train proteins and test proteins
        train_df, test_df = split_dataset(prepped_data, train_prots, test_prots) # Split data according to the protein ids
        print(f'Train dataset shape: {train_df.shape}')
        print(f'Test dataset shape: {test_df.shape}')
        
        test_path = f'./{args.o}/{args.n}_test_data.json'
        save_as_string(list(test_prots), test_path)
        print(f'Test prots saved to {test_path}')
        
        train_dataset = create_dataset(tokenizer=tokenizer, seqs=list(train_df['sequence']), labels=list(train_df['label']),
                                    max_batch_residues=args.batch_size) # Create a huggingface dataset
        test_dataset = create_dataset(tokenizer=tokenizer, seqs=list(test_df['sequence']), labels=list(test_df['label']), 
                                    max_batch_residues=args.batch_size)
    else:
        data = pd.read_json(args.dataset_path)
        train_df, test_df = train_test_split(prepped_data, random_state=args.seed)
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

    model = TokenClassifier(pbert, fine_tune=args.fine_tune)
    #compiled_model = torch.compile(model)
    #compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
    tokenizer, compiled_model = train_model(args, train_ds=train_dataset, test_ds=test_dataset, model=model, tokenizer=tokenizer,
                       seed=args.seed, batch=args.batch_size, val_batch=args.val_batch, epochs=args.epochs, accum=args.accum, lr=args.lr)

    return tokenizer, model

if __name__ == '__main__':
    args = parser.parse_args()
    tokenizer, model, history = main(args)
    now = datetime.now()

    name = args.n

    if not os.path.exists(f'./{args.o}'):
        os.mkdir(f'./{args.o}')

    torch.save(model, f'./{args.o}/{name}.pt')
