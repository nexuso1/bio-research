import json
import pandas as pd
import numpy as np
import os
import baseline

from argparse import ArgumentParser
from utils import load_torch_model, load_tf_model
from train_script import TokenClassifier, create_dataset, load_data
from transformers import BertTokenizer
from sklearn.metrics import f1_score, accuracy_score
from itertools import chain

parser = ArgumentParser()

parser.add_argument('-a', type=bool, help='Analyze mode. Analyze results from an existing result dataframe. The -i argument will then be the dataframe path.', default=False)
parser.add_argument('-i', type=str, help='Model or dataframe path')
parser.add_argument('-t', type=str, help='Test data path', default='./')
parser.add_argument('--prots', type=str, help='Path to protein dataset, mapping IDs to sequences.', default='./phosphosite_sequences/phosphosite_df.json')
parser.add_argument('-p', type=bool, help='Whether the test data are proteins or not', default=True)
parser.add_argument('--max_length', type=int, help='Maximum length of protein sequence to consider (longer sequences will be filtered out of the test data. Default is 1024.', default=1024)
parser.add_argument('--mode', type=str, help='Test mode. Either "pt" for Pytorch models, or "tf" for tensorflow models.', default='tf')

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

def extract_labels_from_row(row, p):
    idxs = [i for i in range(len(row['sequence'])) if row['sequence'][i] == p]
    labels = [row['label'][i] for i in idxs]
    preds = [row['predictions'][i] for i in idxs]
    return labels, preds

def analyze_preds(args, pred_df):
    relevant_prots = ['T', 'S', 'Y', 'R']
    non_canon_prots = ['H', 'C', 'B', 'D', 'N', 'K']
    relevant_preds = [[] for p in relevant_prots]
    relevant_labels = [[] for p in relevant_prots]
    nc_preds = [[] for p in non_canon_prots]
    nc_labels = [[] for p in non_canon_prots]

    for key in pred_df.index:
        row = pred_df.loc[key]
        for i, p in enumerate(relevant_prots):
            labels, preds = extract_labels_from_row(row, p)
            relevant_labels[i].extend(labels)
            relevant_preds[i].extend(preds)
        
        for i, p in enumerate(non_canon_prots):
            labels, preds = extract_labels_from_row(row, p)
            nc_labels[i].extend(labels)
            nc_preds[i].extend(preds)

    print('Usual phosphorylation AA results:')
    for i, p in enumerate(relevant_prots):
        acc = accuracy_score(relevant_labels[i], relevant_preds[i])
        f1 = f1_score(relevant_labels[i], relevant_preds[i], average='macro')
        print(f'AA with FASTA code {p}:')
        print(f'\tAccuracy: {acc}')
        print(f'\tF1: {f1}')

    print('Non-canon phosphorylation AA results:')
    for i, p in enumerate(non_canon_prots):
        acc = accuracy_score(nc_labels[i], nc_preds[i])
        f1 = f1_score(nc_labels[i], nc_preds[i], average='macro')
        print(f'AA with FASTA code {p}:')
        print(f'\tAccuracy: {acc}')
        print(f'\tF1: {f1}')

def flatten_list(lst):
    return list(chain(*lst))

def prepare_prot_df(args, protein_df):
    protein_ids = pd.read_json(args.t, typ='series', orient='records')
    test_df = protein_df.loc[protein_ids]
    test_df = remove_long_sequences(test_df, args.max_length)
    test_df = preprocess_data(test_df)
    
def test_tf_model(args):
    import baseline
    import tensorflow as tf

    model = load_tf_model(args.i)
    data = baseline.load_data(args.t) # tfrec dataset
    protein_df = load_data(args.prots).set_index('id') # protein dataset (dataframe)
    test_data = prepare_prot_df(args, protein_df)

    # Prepare test dataset
    test = data.filter(lambda x: x['uniprot_id'].ref() in test_data.index)
    test = test.batch(256).prefetch(tf.data.AUTOTUNE)

    pred_dict = {id : np.zeros(shape=(len(test_data['sequence'][id]))) for id in test_data.index}
    for batch in test:
        print(batch) 
        preds = model.predict(batch['embeddings'])
        pred_dict[batch['uniprot_id']][batch['position']] = preds.numpy()
        
        break

def calculate_metrics(labels, preds):
    # Calculate metrics
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(f'F1 score: {f1}')
    print(f'Accuracy: {acc}')

def main(args):
    if args.a:
        df = pd.read_json(args.i)
        df['sequence'] = df['sequence'].apply(lambda row: ''.join(row.split()))
        analyze_preds(args, df)
        return

    if args.mode == 'tf':
        test_tf_model(args)
        return

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_torch_model(args.i)
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    protein_df = load_data(args.prots).set_index('id')

    if args.p:
        test_df = prepare_prot_df(args, protein_df)
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
        test_df['sequence'] = protein_df.loc[test_df.index]['sequence'] # Return sequences to their original form
        save_preds(args, test_df)
        
        # Calculate relevant metrics
        calculate_metrics(flatten_list(test_df['label'].to_numpy()), flatten_list(preds))

        # Analyze performance on relevant AAs
        analyze_preds(args, test_df)

    else:
        print('Non-protein data not supported yet. Exiting.')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
