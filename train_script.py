import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import evaluate
import random
import torch.nn.functional as F
import numpy as np
import argparse
import re
from datetime import datetime

from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, BertModel, BertTokenizer, set_seed

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=20)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=2048)
parser.add_argument('--fasta', type=str, help='Path to the FASTA protein database', default='./epsd_sequences/Total.fasta')
parser.add_argument('--phospho', type=str, help='Path to the phoshporylarion dataset', default='./epsd_sequences/Total.txt')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def load_fasta(path : str):
    seq_iterator = SeqIO.parse(open(path), 'fasta')
    seq_dict = {}
    for seq in seq_iterator:
        # extract sequence id
        try:
            seq_id = seq.id.split('|')[0]
        except IndexError:
            # For some reason, some sequences do not contain uniprot ids, so skip them
            continue
        seq_dict[seq_id] = str(seq.seq)

    return seq_dict

def load_phospho(path : str):
    """
    Extracts phosphoryllation site indices from the dataset. 
    Locations expected in the column 'MOD_RSD'.
    
    Returns a dictionary in format {ACC_ID : [list of phosphoryllation site indices]}
    """
    dataset = pd.read_csv(path)
    dataset['position'] = dataset['MOD_RSD'].str.extract(r'[\w]([\d]+)-p')
    grouped = dataset.groupby(dataset['ACC_ID'])
    res = {}
    for id, group in grouped:
        res[id] = group['position'].to_list()
    
    return res

def load_phospho_epsd(path : str):
    data = pd.read_csv(path, sep='\t')
    data.index = data['EPSD ID']
    grouped = data.groupby(data['EPSD ID'])

    res = {}
    for id, group in grouped:
        res[id] = group['Position'].to_list()

    return res

def get_inputs_outputs(fasta_path, phospho_path):
    fasta = load_fasta(fasta_path)
    phospho = load_phospho(phospho_path)

    inputs = []
    targets = []
    for key in phospho.keys():
        inputs.append(fasta[key])
        targets.append(phospho[key])

    return inputs, targets

class ProteinDataset(Dataset):
    def __init__(self,tokenizer, max_length,  
                 inputs : list = None,
                 targets : list = None,
                 fasta_path : str = None,
                 phospho_path : str = None,
                 verbose = 1
                 ) -> None:
        # Load from file if paths given
        if fasta_path and phospho_path:
            self.x, self.y = get_inputs_outputs(fasta_path, phospho_path)
        
        # Take input from given arrays
        else:
            if verbose > 0:
                if inputs is None:
                    
                    print('Warning: No input path given and the inputs parameter is None')
                
                if targets is None:
                    print('Warning: No targets given and the targets parameter is None')

            self.x = inputs
            self.y = targets

        self.verbose = verbose
        self.tokenizer = tokenizer
        self.max_len = max_length

        self.prune_long_sequences()

    def prune_long_sequences(self) -> None:
        """
        Remove all sequences that are longer than self.max_len from the dataset.
        Updates the self.x and self.y attributes.
        """
        keep_x = []
        keep_y = []

        count = 0

        for i in range(len(self.x)):
            if len(self.x[i]) > self.max_len:
                count += 1
                continue

            keep_x.append(self.x[i])
            keep_y.append(self.y[i])

        if self.verbose > 0:
            print(f"Removed {count} sequences from the dataset longer than {self.max_len}.")

        self.x = keep_x
        self.y = keep_y

    def prep_seq(self, seq):
        """
        Prepares the given sequence for the model by subbing rare AAs for X and adding 
        padding between AAs. Required by the base model.
        """
        return " ".join(list(re.sub(r"[UZOB]", "X", seq)))

    def prep_target(self, enc, target):
        """
        Transforms the target into an array of ones and zeros with the same length as the 
        corresponding FASTA protein sequence. Value of one represents a phosphorylation 
        site being present at the i-th AA in the protein sequence.
        """
        res = torch.zeros(self.max_len).long()
        res[target] = 1
        res = res.roll(1)
        for i, idx in enumerate(enc.input_ids.flatten().int()):
            if idx == 0: # [PAD]
                break

            if idx == 2 or idx == 3: # [CLS] or [SEP]
                res[i] = -100 # This label will be ignored by the loss
        return res

    def __getitem__(self, index):
        seq = self.x[index]
        target =self.y[index]
        seq = self.prep_seq(seq)
        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length = self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        target = self.prep_target(encoding, target)
        return {
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'labels' : target
        }

    def __len__(self):
        return len(self.x)

class ProteinEmbed(nn.Module):
    def __init__(self, base_model : nn.Module, dropout = 0.2, n_labels = 2, transfer_learning=True) -> None:
        super(ProteinEmbed, self).__init__()
        self.base = base_model
        self.n_labels = n_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.base.config.hidden_size, self.n_labels)
        self.init_weights()

        if transfer_learning:
            self.freeze_base()

    def init_weights(self):
        torch.nn.init.normal_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)
        
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
        return_dict = None,
    ):
        outputs = self.base(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss' : loss,
            'logits' : logits,
            'hidden_states' : outputs.hidden_states,
            'attentions' : outputs.attentions,
            'outputs' : (loss, outputs)
        }

    
def get_bert_model():
    pbert = BertModel.from_pretrained("Rostlab/prot_bert")
    pbert.config.num_labels = 2
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    return pbert, tokenizer

def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

def compute_metrics(eval_pred, metric):
    preds, labels = eval_pred
    return metric.compute(predictions = preds, references=labels)

def train_model(train_ds, test_ds, model, tokenizer,
                lr=3e-4, epochs=1, batch=50, val_batch=100, accum=1, seed=42, deepspeed=None):

    # Set all random seeds
    set_seeds(seed)

    # Huggingface Trainer arguments
    args = TrainingArguments(
        "./",
        evaluation_strategy = "epoch",
        logging_strategy = "epoch",
        save_strategy = "no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed = seed
    )
    
    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    return tokenizer, model, trainer.state.log_history

def main(args):
    inputs, outputs = get_inputs_outputs(args.fasta, args.phospho)
    pbert, tokenizer = get_bert_model()
    train_X, test_X, train_y, test_y = train_test_split(inputs, outputs, random_state=args.seed)
    model = ProteinEmbed(pbert)
    model = torch.compile(model)
    model.to(device)
    train_dataset = ProteinDataset(tokenizer=tokenizer, max_length=args.max_length, inputs=train_X, targets=train_y)
    test_dataset = ProteinDataset(tokenizer=tokenizer, max_length=args.max_length, inputs=test_X, targets=test_y)

    return train_model(train_ds=train_dataset, test_ds=test_dataset, model=model, tokenizer=tokenizer, seed=args.seed,
                       batch=args.batch_size, epochs=args.epochs)

if __name__ == '__main__':
    args = parser.parse_args()
    tokenizer, model, history = main(args)
    now = datetime.now()

    name = now.strftime("model_%Y-%M-%d_%H:%M:%S")
    torch.save(model, f'./output/{name}')
