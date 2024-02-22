import torch
import json
import pandas as pd
import numpy as np
import os

from argparse import ArgumentParser
from utils import load_torch_model
from train_script import TokenClassifier, create_dataset, load_data
from transformers import BertTokenizer
from sklearn.metrics import f1_score, accuracy_score
from itertools import chain

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()

parser.add_argument('-i', type=str, help='Model path')
parser.add_argument('-t', type=str, help='Test data path')
parser.add_argument('--prots', type=str, help='Path to protein dataset, mapping IDs to sequences.', default='./phosphosite_sequences/phosphosite_df.json')
parser.add_argument('-p', type=bool, help='Whether the test data are proteins or not', default=True)
parser.add_argument('--max_length', type=int, help='Maximum length of protein sequence to consider (longer sequences will be filtered out of the test data. Default is 1024.', default=1024)

def preprocess_data(df : pd.DataFrame):
    """
    Preprocessing for Pbert/ProtT5
    """
    df['sequence'] = df['sequence'].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    df['sequence'] = df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    return df

def remove_long_sequences(df, max_length):
    mask = df['sequence'].apply(lambda x: len(x) < max_length)
    return df[mask]

def save_preds(args, pred_df):
    model_name = os.path.basename(args.i)
    if not os.path.exists('./output/'):
        os.mkdir('./output')
    path = f'./output/{model_name}_preds.json'
    pred_df.to_json(path)
    print(f'Results saved to {path}')

def flatten_list(lst):
    return list(chain(*lst))

def main(args):
    model = load_torch_model(args.i)
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    protein_df = load_data(args.prots).set_index('id')

    if args.p:
        protein_ids = pd.read_json(args.t, typ='series', orient='records')
        test_df = protein_df.loc[protein_ids]
        test_df = remove_long_sequences(test_df, args.max_length)
        test_df = preprocess_data(test_df)
        preds = []
        probs = []
        with torch.no_grad():
            for i, prot in enumerate(test_df.index):
                if i % 100 == 0:
                    print(i)
                inputs = tokenizer(test_df.loc[prot]['sequence'], padding=False, return_tensors='pt').to(device)
                logits = model(**inputs)[0].to('cpu')
                prob = torch.nn.functional.softmax(logits, dim=-1).numpy()
                pred = np.argmax(prob, axis=-1)
                preds.append(pred.flatten()[1:-1]) # Slice off the padding tokens
                probs.append(prob[0][1:-1, :])
        
        # Save the predictions for later inspection
        test_df['probabilities'] = probs
        test_df['predictions'] = preds
        save_preds(args, test_df)

        # Calculate metrics
        np_labels = flatten_list(test_df['label'].to_numpy())
        np_preds = flatten_list(preds)
        print(len(np_labels) == len(np_preds))
        print(np_preds)
        f1 = f1_score(np_labels, np_preds, average='macro')
        acc = accuracy_score(np_labels, np_preds)
        print(f'F1 score: {f1}')
        print(f'Accuracy: {acc}')

    else:
        print('Non-protein data not supported yet. Exiting.')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
