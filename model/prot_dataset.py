import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset

class ProteinDataset(Dataset):
    def __init__(self, data : pd.DataFrame) -> None:
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index : int):
        row = self.data.iloc[index]
        return index, row.sequence, row.label
    
class FullProteinDataset:
    def __init__(self, prot_info : pd.DataFrame, splits : dict, tokenizer=None, pre_tokenize=False):
        self.prot_info = prot_info
        self.split_info = splits
        self.n_splits = len(splits)

        if pre_tokenize:
            if self.tokenizer is not None:
                self.tokenizer = tokenizer
                self.tokenize_datasets()
            else:
                raise ValueError('No tokenizer provided for pre-tokenization.')
    
    def tokenize_datasets(self):
        ...

    def get_fold(self, i):
        test = self.prot_info.loc[self.split_info[i]['test']]
        train = self.prot_info.loc[self.split_info[i]['train']]
        test_ds = ProteinDataset(test)
        train, dev = train_test_split(train, train_size=0.8, random_state=42)
        train_ds = ProteinDataset(train)
        dev_ds = ProteinDataset(dev)

        print(f'Train size: {len(train_ds)}')
        print(f'Dev size: {len(dev_ds)}')
        print(f'Test size: {len(test_ds)}')

        return train_ds, dev_ds, test_ds