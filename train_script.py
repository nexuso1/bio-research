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

from datetime import datetime
from utils import remove_long_sequences, load_prot_data
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, BertModel, BertTokenizer, set_seed, DataCollatorForTokenClassification

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size', default=3)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=10)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=1024)
parser.add_argument('--fasta', type=str, help='Path to the FASTA protein database', default='./phosphosite_sequences/Phosphosite_seq.fasta')
parser.add_argument('--phospho', type=str, help='Path to the phoshporylarion dataset', default='./phosphosite_sequences/Phosphorylation_site_dataset')
parser.add_argument('--dataset_path', type=str, help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.', default='./phosphosite_sequences/phosphosite_df.json')
parser.add_argument('--pretokenized', type=bool, help='Input dataset is already pretokenized', default=False)
parser.add_argument('--val_batch', type=int, help='Validation batch size', default=4)
parser.add_argument('--clusters', type=str, help='Path to clusters', default='cluster30.tsv')
parser.add_argument('--fine_tune', type=bool, help='Use fine tuning on the base model or not. Default is False', default=False)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=3)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-5)
parser.add_argument('-o', type=str, help='Output folder', default='output')
parser.add_argument('-n', type=str, help='Model name', default='prot_model.pt')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.classifier = nn.Sequential(
            nn.Linear(self.base.config.hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_labels)
        )

        if not fine_tune:
            self.freeze_base()
        
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
    pbert.config.num_labels = 2
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    return pbert, tokenizer

def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    f1 = evaluate.load('f1')
    return f1.compute(predictions = preds, references=labels)

def create_dataset(tokenizer, seqs, labels, max_length):
    tokenized = tokenizer(seqs, max_length=max_length, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    # we need to cut of labels after max_length positions for the data collator to add the correct padding ((max_length - 1) + 1 special tokens)
    labels = [l[: max_length - 1] for l in labels] 
    dataset = dataset.add_column("labels", labels)
     
    return dataset

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

def train_model(args, train_ds, test_ds, model, tokenizer,
                lr, epochs, batch, val_batch, accum, seed=42, deepspeed=None):

    # Set all random seeds
    set_seeds(seed)

    # Huggingface Trainer arguments
    args = TrainingArguments(
        evaluation_strategy = "no",
        logging_strategy = "epoch",
        save_strategy = "epoch",
        output_dir = f"/storage/praha1/home/nexuso1/bio-research/temp_output/{args.n}",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed = seed,
        remove_unused_columns=True,
        eval_accumulation_steps=10,
        weight_decay=0.001,
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
                                    max_length=args.max_length) # Create a huggingface dataset
        test_dataset = create_dataset(tokenizer=tokenizer, seqs=list(test_df['sequence']), labels=list(test_df['label']), 
                                    max_length=args.max_length)
    else:
        data = pd.read_json(args.dataset_path)
        train_df, test_df = train_test_split(prepped_data, random_state=args.seed)
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

    model = TokenClassifier(pbert, fine_tune=args.fine_tune)
    compiled_model = torch.compile(model)
    compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
    tokenizer, compiled_model, history = train_model(args, train_ds=train_dataset, test_ds=test_dataset, model=compiled_model, tokenizer=tokenizer,
                       seed=args.seed, batch=args.batch_size, val_batch=args.val_batch, epochs=args.epochs, accum=args.accum, lr=args.lr)

    return tokenizer, model, history

if __name__ == '__main__':
    args = parser.parse_args()
    tokenizer, model, history = main(args)
    now = datetime.now()

    name = args.n

    if not os.path.exists(f'./{args.o}'):
        os.mkdir(f'./{args.o}')

    torch.save(model, f'./{args.o}/{name}.pt')
