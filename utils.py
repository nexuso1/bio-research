import pandas as pd
import numpy as np
import re

from Bio import SeqIO
from datasets import Dataset

def load_torch_model(path):
    import torch

    with open(path, 'rb') as f:
        model = torch.load(f)

    return model

def load_prot_data(dataset_path):
    df = pd.read_json(dataset_path)
    df = df.dropna()
    df['sites'] = df['sites'].apply(lambda x: [eval(i) - 1 for i in x])
    labels = [np.zeros(shape=len(s)) for s in df['sequence']]
    for i, l in enumerate(labels):
        l[df.iloc[i]['sites']] = 1

    df['label'] = labels
    
    return df[['id', 'sequence', 'label']]

def load_tf_model(path):
    import tensorflow as tf
    
    return tf.keras.models.load_model(path)

def load_clusters(path):
    return pd.read_csv(path, sep='\t', names=['cluster_rep', 'cluster_mem'])

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

def remove_long_sequences(df, max_length):
    mask = df['sequence'].apply(lambda x: len(x) < max_length)
    return df[mask]

class ProteinDataset(Dataset):
    def __init__(self,tokenizer, max_length,  
                 inputs : list = None,
                 targets : list = None,
                 verbose = 1
                 ) -> None:
        
        # Take input from given arrays
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

    def load_data(self, dataset_path):
        df = pd.read_json(dataset_path)
        df = df.dropna()
        df['sites'] = df['sites'].apply(lambda x: [eval(i) - 1 for i in x])
        labels = [np.zeros(shape=len(s)) for s in df['sequence']]
        for i, l in enumerate(labels):
            l[df.iloc[i]['sites']] = 1

        df['label'] = labels
    
        return df[['id', 'sequence', 'label']]

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
        res = np.zeros(self.max_len)
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
