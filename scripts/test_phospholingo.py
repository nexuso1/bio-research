import torchmetrics
import torch
import sys

sys.path.append('.')
sys.path.append('./model')
sys.path.append('./data/phospho_lingo/')

from tqdm import tqdm
from argparse import ArgumentParser
from model.data_loading import load_phoshpolingo_dataset
from model.utils import load_torch_model
from model.esm import load_from_checkpoint, compute_metrics, get_esm
from model.classifiers import RNNTokenClassifer
import os
import pandas as pd

parser = ArgumentParser()

parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--dataset_path', type=str, default='')
parser.add_argument('--dataset_type', type=str, default='test', help='Dataset is either "train", "valid" or "test"')
parser.add_argument('--chkpt', action='store_true', default=False, help='Model is a checkpoint')
parser.add_argument('--esm_type', type=str, default='3B')
def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.chkpt:
        model = load_from_checkpoint(args.i)
    else:
        saved_model = load_torch_model(args.model_path)
        state_dict, config = saved_model['state_dict'], saved_model['config']
        esm, tokenizer = get_esm(args.esm_type)
        model = RNNTokenClassifer(config, esm)
        model.load_state_dict(state_dict)

    model.eval()
    dev = load_phoshpolingo_dataset(args.dataset_path, args.dataset_type, 4,
                                     tokenizer=tokenizer, num_workers=16, shuffle=False)
    
    metrics = {
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=-1).to(device),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=-1).to(device),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=-1).to(device),
    }
    loss_metric =  torchmetrics.MeanMetric().to(device)
    roc = torchmetrics.ROC(task='binary', ignore_index=-1).to(device)
    prc = torchmetrics.PrecisionRecallCurve(task='binary', ignore_index=-1).to(device)
    metrics = torchmetrics.MetricCollection(metrics)

    if args.p:
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
                mask = batch['labels'].view(-1) != -1
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

        # Save the predictions for later inspection

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

        # # Rest of probabilities
        # dev_df.set_index('id')
        # dev_df['probabilities'] = probs
        # dev_df['predictions'] = preds
        # dev_df['sequence'] = protein_df.loc[dev_df.index]['sequence'] # Return sequences to their original form
        # save_preds(args, dev_df)
        
        # # Calculate relevant metrics
        # # calculate_metrics(flatten_list(test_df['label'].to_numpy()), flatten_list(preds))

        # # Analyze performance on relevant AAs
        # analyze_preds(args, dev_df)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)