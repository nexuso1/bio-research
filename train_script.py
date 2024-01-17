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

from datasets import Dataset
from torch.nn import CrossEntropyLoss
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, BertModel, BertTokenizer, set_seed, DataCollatorForTokenClassification

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size', default=1)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=1)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=2048)
parser.add_argument('--fasta', type=str, help='Path to the FASTA protein database', default='./phosphosite_sequences/Phosphosite_seq.fasta')
parser.add_argument('--phospho', type=str, help='Path to the phoshporylarion dataset', default='./phosphosite_sequences/Phosphorylation_site_dataset')
parser.add_argument('--dataset_path', type=str, help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.', default='./phosphosite_sequences/phosphosite_df.json')

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
    dataset = pd.read_csv(path, sep='\t', skiprows=3)
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

def get_inputs_outputs(dataset_path):
    df = pd.read_json(dataset_path)
    df = df.dropna()
    df['sites'] = df['sites'].apply(lambda x: [eval(i) - 1 for i in x])
    labels = [np.zeros(shape=len(s)) for s in df['sequence']]
    for i, l in enumerate(labels):
        l[df.iloc[i]['sites']] = 1

    df['label'] = labels
    
    return df[['sequence', 'label']]

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
        self.prep_data()

    def prune_long_sequences(self) -> None:
        """
        Remove all sequences that are longer than self.max_len from the dataset.
        Updates the self.x and self.y attributes.
        """
        mask = self.x.apply(lambda x: len(x) < self.max_len)
        new_x = self.x[mask]
        new_y = self.y[mask]

        if self.verbose > 0:
            count = self.x.shape[0] - new_x.shape[0]
            print(f"Removed {count} sequences from the dataset longer than {self.max_len}.")

        self.x = new_x
        self.y = new_y

    def prep_seq(self, seq):
        """
        Prepares the given sequence for the model by subbing rare AAs for X and adding 
        padding between AAs. Required by the base model.
        """
        # print(seq)
        cleaned = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
        return cleaned
    
    def prep_data(self):
        prepped = np.array(self.x.apply(self.prep_seq), dtype=np.int32)
        tokenized = self.tokenizer(prepped)
        targets = [self.prep_target(tokenized.iloc[i], self.y.iloc[i]) for i in range(self.y.shape[0])]
        self.data = tokenized
        self.targets = targets

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
        encoding = self.data.iloc[index]
        target =self.y[index]
        return {
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'labels' : target
        }

    def __len__(self):
        return self.x.shape[0]

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

def create_dataset(tokenizer, seqs, labels, max_length):
    tokenized = tokenizer(seqs, max_length=max_length, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    # we need to cut of labels after max_length positions for the data collator to add the correct padding ((max_length - 1) + 1 special tokens)
    labels = [l[: max_length - 1] for l in labels] 
    dataset = dataset.add_column("labels", labels)
     
    return dataset

def train_model(train_ds, test_ds, model, tokenizer,
                lr=3e-4, epochs=1, batch=50, val_batch=100, accum=1, seed=42, deepspeed=None):

    # Set all random seeds
    set_seeds(seed)

    # Huggingface Trainer arguments
    args = TrainingArguments(
        "./",
        evaluation_strategy = "steps",
        eval_steps=10,
        logging_strategy = "epoch",
        save_strategy = "no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed = seed
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer, 
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    return tokenizer, model, trainer.state.log_history

def preprocess_data(df : pd.DataFrame):
    df['sequence'] = df['sequence'].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    df['sequence'] = df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    return df

def main(args):
    data = get_inputs_outputs(args.dataset_path)
    prepped_data = preprocess_data(data)
    pbert, tokenizer = get_bert_model()
    train_df, test_df = train_test_split(prepped_data, random_state=args.seed)
    model = ProteinEmbed(pbert)
    #model = torch.compile(model)
    model.to(device)
    train_dataset = create_dataset(tokenizer=tokenizer, seqs=list(train_df['sequence']), labels=list(train_df['label']),
                                   max_length=args.max_length)
    test_dataset = create_dataset(tokenizer=tokenizer, seqs=list(test_df['sequence']), labels=list(test_df['label']), 
                                  max_length=args.max_length)

    return train_model(train_ds=train_dataset, test_ds=test_dataset, model=model, tokenizer=tokenizer,
                       seed=args.seed, batch=args.batch_size, epochs=args.epochs)

if __name__ == '__main__':
    args = parser.parse_args()
    tokenizer, model, history = main(args)
    now = datetime.now()

    name = now.strftime("model_%Y-%M-%d_%H:%M:%S")
    torch.save(model, f'./output/{name}')
