import torch
import json
import pandas as pd

from argparse import ArgumentParser
from utils import load_torch_model

parser = ArgumentParser()

parser.add_argument('-i', type=str, help='Model path')
parser.add_argument('-t', type=str, help='Test data path')
parser.add_argument('--prots', type=str, help='Path to protein dataset, mapping IDs to sequences.', default='./phosphosite_sequences/phosphosite_df.json')
parser.add_argument('-p', type=bool, help='Whether the test data are proteins or not', default=True)

def preprocess_data(df : pd.DataFrame):
    """
    Preprocessing for Pbert/ProtT5
    """
    df['sequence'] = df['sequence'].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    df['sequence'] = df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    return df

def main(args):
    model = load_torch_model(args.i)
    protein_df = pd.read_json(args.prots).set_index('id')
    protein_df = preprocess_data(protein_df)
    if args.p:
        with open(args.i, 'r') as f:
            protein_ids = json.load(f)

        seqs = protein_df[protein_ids]['sequence']
        sites = protein_df[protein_ids]['sites']
        with torch.no_grad():
            logits = model(seqs)
            preds = torch.argmax(torch.softmax(logits), -1)
            print(preds)
        
    else:
        print('Non-protein data not supported yet. Exiting.')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)