from torch.utils.data.dataset import Dataset
import pandas as pd

class ProteinTorchDataset(Dataset):
    def __init__(self, data : pd.DataFrame) -> None:
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index : int):
        row = self.data.iloc[index]
        return row.sequence, row.label
    
class PhosphoLingoDataset(Dataset):
    def __init__(self, data : list[dict]) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['prot_id'], item['seq_data'], item['labels']
