import os
import pandas as pd
import numpy as np
import torchmetrics
import torch
import matplotlib.pyplot as plt


from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from model.esm import compute_metrics, get_esm
from model.classifiers import RNNClassifier
from argparse import ArgumentParser
from model.utils import load_torch_model, preprocess_data
from model.data_loading import remove_long_sequences, prepare_datasets, load_prot_data
from model.prot_dataset import ProteinTorchDataset
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

parser = ArgumentParser()

parser.add_argument('-a', type=bool, help='Analyze mode. Analyze results from an existing result dataframe. The -i argument will then be the dataframe path.', default=False)
parser.add_argument('--i', type=str, help='Model or dataframe path', default='logs\esm.py-2024-05-10_210142\esm.pt.pt')
parser.add_argument('--test_clusters', type=str, help='Test cluster path', default='output\esm.pt_test_data.json')
parser.add_argument('--prots', type=str, help='Path to protein dataset, mapping IDs to sequences.', default='./phosphosite_sequences/phosphosite_df.json')
parser.add_argument('--max_length', type=int, help='Maximum length of protein sequence to consider (longer sequences will be filtered out of the test data. Default is 1024.', default=1024)
parser.add_argument('--chkpt', action='store_true', default=False, help='Model is a checkpoint')
parser.add_argument('--batch_size', default=8, help='Batch size', type=int)
parser.add_argument('--num_workers', default=0, type=int, help='Num parallel workers')
parser.add_argument('--prot_info_path', type=str, 
                     help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.',
                     default='data/phosphosite_sequences/phosphosite_df_small.json')
parser.add_argument('--train_path', type=str, help='Path to train protein IDs, subset of IDs in the prot. info dataset. JSON list.',
                    default='data/cleaned_train_prots.json')
parser.add_argument('--test_path', type=str, help='Path to test protein IDs, subset of IDs in the prot. info dataset. JSON list.',
                    default='data/cleaned_test_prots.json')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_preds(args, pred_df):
    model_name = os.path.basename(args.i)
    if not os.path.exists('./output/'):
        os.mkdir('./output')
    path = f'./output/{model_name}_preds.json'
    pred_df.to_json(path)
    print(f'Results saved to {path}')

def extract_labels_from_row(row, p):
    #idxs = [i for i in range(len(row['sequence'])) if row['sequence'][i] == p]
    mask = row['sequence'] == p
    # labels = [row['label'][i] for i in idxs]
    labels = row['label'][mask]
    #preds = [row['predictions'][i] for i in idxs]
    preds = row['predictions'][mask]
    return labels, preds

def analyze_preds(args, pred_df):
    relevant_prots = ['T', 'S', 'Y']
    non_canon_prots = ['H', 'C', 'B','R','D', 'N', 'K']
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
        f1 = f1_score(relevant_labels[i], relevant_preds[i])
        cm = confusion_matrix(relevant_labels[i], relevant_preds[i])
        print(f'AA with FASTA code {p}:')
        print(f'\tAccuracy: {acc}')
        print(f'\tF1: {f1}')
        print(f'\tConfusion matrix: {cm}')
        print(f'\tNumber of predictions: {len(relevant_preds[i])}')
        print(f'\tNumber of true labels: {len(relevant_labels[i])}')

    print('Non-canon phosphorylation AA results:')
    for i, p in enumerate(non_canon_prots):
        acc = accuracy_score(nc_labels[i], nc_preds[i])
        f1 = f1_score(nc_labels[i], nc_preds[i])
        cm = confusion_matrix(relevant_labels[i], relevant_preds[i])
        print(f'AA with FASTA code {p}:')
        print(f'\tAccuracy: {acc}')
        print(f'\tConfusion matrix: {cm}')
        print(f'\tF1: {f1}')
        print(f'\tNumber of predictions: {len(nc_preds[i])}')
        print(f'\tNumber of true labels: {len(nc_labels[i])}')

def prepare_prot_df(args, protein_df):
    protein_ids = pd.read_json(args.t, typ='series', orient='records')
    test_df = protein_df.loc[protein_ids]
    test_df = remove_long_sequences(test_df, args.max_length)
    test_df = preprocess_data(test_df)
    return test_df

def calculate_metrics(labels, preds):
    # Calculate metrics
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    print(f'F1 score: {f1}')
    print(f'Accuracy: {acc}')

def load_from_checkpoint(path, fine_tune=False, use_lora=False):
    """
    Loads the model checkpoint from the given path. Creates a TokenClassifier instance, 
    with and passes it args from the checkpoint, and flags from the from the arguments.
    The created model loads the state dict from the checkpoint. Also creates an optimizer with 
    a state_dict from the checkpoint. 
    
    Returns a quintuple (model, opitm, epoch, loss, args)
    """
    chkpt = torch.load(path)
    args, epoch, loss = chkpt['args'], chkpt['epoch'], chkpt['loss']
    base, tokenizer = get_esm(args)
    model = RNNClassifier(args, base, fine_tune=fine_tune, use_lora=use_lora)
    model.load_state_dict(chkpt['model_state_dict'])
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay)
    optim.load_state_dict(chkpt['optimizer_state_dict'])
    return model

def main(args):
    if args.a:
        df = pd.read_json(args.i)
        df['sequence'] = df['sequence'].apply(lambda row: ''.join(row.split()))
        analyze_preds(args, df)
        return

    if args.chkpt:
        model = load_from_checkpoint(args.i)
    else:
        model = load_torch_model(args.i)
    
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    prot_info = load_prot_data(args.prot_info_path)
    _, dev, _, dev_dataset = prepare_datasets(args, tokenizer, ignore_label=-1, return_datasets=True)
    dev_df = dev_dataset.data 
    
    metrics = {
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=-100).to(device),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=-100).to(device),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=-100).to(device),
    }
    loss_metric =  torchmetrics.MeanMetric().to(device)
    roc = torchmetrics.ROC(task='binary', ignore_index=-100).to(device)
    prc = torchmetrics.PrecisionRecallCurve(task='binary', ignore_index=-100).to(device)
    metrics = torchmetrics.MetricCollection(metrics)

    preds_list = []
    probs = []
    
    model.eval()
    metrics.reset()

    epoch_message = f""
    progress_bar = tqdm(range(len(dev)))
    with torch.no_grad():
        for batch in dev:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Model returns a tuple, logits are the first element when not given labels
            loss, logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], batch_lens=batch['batch_lens'], labels=batch['labels'])
            mask = batch['labels'].view(-1) != -100
            preds = torch.sigmoid(logits.view(-1)[mask])
            target = batch['labels'].view(-1)[mask]
            logs = compute_metrics(preds.view(-1, 1), target, metrics)
            loss_metric.update(loss)
            roc.update(preds, target.long())
            prc.update(preds, target.long())

            logs['loss'] = loss_metric.compute()
            message = [epoch_message] + [
                f"dev_{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}"
                for k, v in logs.items()
            ]
            progress_bar.set_description(" ".join(message))
            progress_bar.update(1)  
            preds_list.extend(list((preds > 0.5).cpu().numpy().astype('int'))) # Predicted labels
            probs.extend(list(preds.cpu().numpy()))



    # ROC and PRC computation
    for metric, name in [(roc, 'roc'), (prc, 'prc')]:
        fig, ax = metric.plot(score=True)
        fig.savefig(os.path.join(os.path.dirname(args.i), f'{name}.png'))
        fpr, tpr, thresholds = metric.compute()
        if thresholds.shape[0] < tpr.shape[0]:
            thresholds = torch.concatenate([thresholds, torch.Tensor([1]).to(device)], -1) # Last threshold is missing 
        df = pd.DataFrame.from_dict({
            'fpr' : fpr.cpu().numpy(),
            'tpr' : tpr.cpu().numpy(),
            'threshold' : thresholds.cpu().numpy()
        }, orient='columns')
        df.to_json(os.path.join(os.path.dirname(args.i), f'{name}_df.json'), indent=4)

    # Rest of probabilities
    dev_df.set_index('id')
    dev_df['probabilities'] = probs
    dev_df['predictions'] = preds
    dev_df['sequence'] = prot_info.loc[dev_df.index]['sequence'] # Return sequences to their original form

    # Save the predictions for later inspection
    save_preds(args, dev_df)
    
    # Calculate relevant metrics
    # calculate_metrics(flatten_list(test_df['label'].to_numpy()), flatten_list(preds))

    # Analyze performance on relevant AAs
    analyze_preds(args, dev_df)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
