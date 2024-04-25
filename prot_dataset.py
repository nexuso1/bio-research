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
    
