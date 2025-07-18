# This file contains various utility functions and classes used throughout other parts of the project.

import pandas as pd
import numpy as np
import re
import os
import json
import torch

from torch._C import _log_api_usage_once
from torch.nn.functional import binary_cross_entropy_with_logits
from itertools import chain
from Bio import SeqIO
from datasets import Dataset



def load_torch_model(path):
    import torch
    with open(path, 'rb') as f:
        model = torch.load(f)

    return model

def flatten_list(lst):
    return list(chain(*lst))

def preprocess_data(df : pd.DataFrame):
    """
    Preprocessing for Pbert/ProtT5
    """
    df['sequence'] = df['sequence'].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    df['sequence'] = df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    return df

def save_as_string(obj, path):
    """
    Saves the given object as a JSON string.
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, 'w') as f:
        json.dump(obj, f)

class ProteinDataset(Dataset):
    def __init__(self,tokenizer, max_length,  
                 inputs : list = None,
                 targets : list = None,
                 verbose = 1
                 ) -> None:
        

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
        self.input_ids = tokenized['input_ids']
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


class SimpleNamespace:
    def __init__(self, **kwargs) -> None:
        if len(kwargs.keys()) > 0:
            for k, v in kwargs.items():
                self.__setattr__(k, v)

class Metadata(SimpleNamespace):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if "data" not in self.__dict__.keys():
            self.data = {}

    def jsonify_fn(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return obj.__dict__
    
    def save(self, dir : str):
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'metadata.json'), 'w') as f:
            json.dump(self,f, default=self.jsonify_fn,  sort_keys=True, indent=4 )
            
# Taken from torchvision source
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = torch.sigmoid(inputs)
    ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss
