import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset

class ProteinDataset(Dataset):
    def __init__(self, data : pd.DataFrame) -> None:
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index : int):
        row = self.data.loc[index]
        return index, row.sequence, row.label
    
class FullProteinDataset:
    def __init__(self, prot_info : pd.DataFrame, split_info : dict, tokenizer=None, pre_tokenize=False):
        self.prot_info = prot_info
        self.split_info = split_info
        self.n_splits = len(split_info['splits'])
        self.create_datasets()

        if pre_tokenize:
            if self.tokenizer is not None:
                self.tokenizer = tokenizer
                self.tokenize_datasets()
            else:
                raise ValueError('No tokenizer provided for pre-tokenization.')
    
    def tokenize_datasets(self):
        ...

    def create_datasets(self):
        reindexed = self.prot_info.set_index('id')
        test = reindexed.loc[self.split_info['test']]
        train = reindexed.loc[self.split_info['train']]
        self.train_df = train
        self.test_ds = ProteinDataset(test)

        print(f'Test size: {len(self.test_ds)}')
        print(f'Average fold train size: {sum([len(self.split_info["splits"][i]["train"]) for i in range(self.n_splits)]) / self.n_splits}')
        print(f'Average fold dev size: {sum([len(self.split_info["splits"][i]["dev"]) for i in range(self.n_splits)]) / self.n_splits}')
    
    def get_fold(self, i):
        train_prots = np.array(self.split_info['train'])
        split = self.split_info['splits'][i]
        train_indices = split['train']
        dev_indices = split['dev']
        split_train, split_dev = train_prots[train_indices], train_prots[dev_indices]

        return ProteinDataset(self.train_df.loc[split_train].reset_index(level='id')), \
               ProteinDataset(self.train_df.loc[split_dev].reset_index(level='id'))
        
class PhosphoLingoDataset(Dataset):
    def __init__(self, data : list[dict]) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['prot_id'], item['seq_data'], item['labels']
