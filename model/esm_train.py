import numpy as np
import torch
import random
import pandas as pd
import argparse
import json
import os

import torch.utils
import torch.utils.data
import torchmetrics
import datetime
import lora

from functools import partial
from dataclasses import dataclass
from tqdm.auto import tqdm
from utils import Metadata, load_torch_model, sigmoid_focal_loss
from data_loading import prepare_datasets
from datasets import Dataset
from transformers import set_seed, EsmModel, AutoTokenizer
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from classifiers import RNNTokenClassifier, RNNTokenClassiferConfig
from training import device, parser,save_model

print(device)

@dataclass
class TrainingConfig:
    epochs : int
    args : argparse.Namespace
    accum : int
    batch_size : int
    weight_decay : float
    lr : float
    logdir : str
    metadata : Metadata=None
    seed : int = 42
    start_epoch : int = 0
    f1_min : int = 0
    optim : torch.optim.Optimizer = None

def get_esm(type):
    if type == '3B':
        model, tokenizer = EsmModel.from_pretrained('facebook/esm2_t36_3B_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
    elif type == '15B':
        model, tokenizer = EsmModel.from_pretrained('facebook/esm2_t48_15B_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')
    elif type == '35M':
        model, tokenizer = EsmModel.from_pretrained('facebook/esm2_t12_35M_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    else:
        model, tokenizer = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D'), AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    return model, tokenizer

def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

def save_preds(path, preds : list):
    pd.Series(preds).to_json(path)
    print(f'Predictions saved to {path}.')

@torch.no_grad()
def eval_model(model, test_ds, epoch, metrics : torchmetrics.MetricCollection):
    model.eval()
    metrics.reset()
    loss_metric =  torchmetrics.MeanMetric().to(device)
    epoch_message = f"Epoch={epoch+1}"
    progress_bar = tqdm(range(len(test_ds)))
    probs_list = []
    with torch.no_grad():
        for batch in test_ds:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model.predict(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], batch_lens=batch['batch_lens'], labels=batch['labels'])
            mask = batch['labels'].view(-1) != model.ignore_index
            preds = torch.sigmoid(logits.view(-1)[mask])
            target = batch['labels'].view(-1)[mask]
            logs = compute_metrics(preds.view(-1, 1), target, metrics)
            loss_metric.update(loss)
            logs['loss'] = loss_metric.compute()
            message = [epoch_message] + [
                f"dev_{k}={v:.{0<abs(v)<2e-4 and '3g' or '4f'}}"
                for k, v in logs.items()
            ]
            progress_bar.set_description(" ".join(message))
            progress_bar.update(1)
            # Extract the predicitions
            valid_logits = logits[batch['labels'] != model.ignore_index] # Gather valid logits
            indices = np.cumsum(batch['batch_lens'].cpu().numpy(), 0) # Indices into gathered logits according to batch lengths
            probs = np.split(valid_logits.cpu().numpy(), indices)[:-1] # Split them according to the batch lenghts, last element is extra
            probs_list.extend(probs) # Predicted probabilites
    return {k : v.cpu().numpy() for k, v in logs.items()}, probs_list

def train_model(train_ds : Dataset, dev_ds : Dataset, model : torch.nn.Module, config : TrainingConfig):

    # Set all random seeds
    set_seeds(config.seed)

    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) if config.optim is None else config.optim

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, len(train_ds) * config.epochs)

    metrics = {
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=model.ignore_index).to(device),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=model.ignore_index).to(device),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=model.ignore_index).to(device),
        'auroc' : torchmetrics.AUROC('binary', ignore_index=model.ignore_index).to(device),
        'mcc' : torchmetrics.MatthewsCorrCoef('binary', ignore_index=model.ignore_index).to(device)
    }
    loss_metric =  torchmetrics.MeanMetric().to(device)
    metrics = torchmetrics.MetricCollection(metrics)

    history = []
    best_f1 = config.f1_min
    # Train model
    for epoch in range(config.start_epoch, config.epochs):
        model.train()
        metrics.reset()
        epoch_message = f"Epoch={epoch+1}/{config.epochs}"
        # Progress bar
        data_and_progress = tqdm(
            train_ds,
            epoch_message,
            unit="batch",
            leave=False,
        )

        for i, batch in enumerate(train_ds):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model.train_predict(**batch)

            # Normalize the loss by the number of accumulation steps
            loss = loss / config.accum

            if config.accum == 1 or ( i > 0 and i % config.accum == 0) or (i + 1 == len(train_ds)):
                loss.backward()
                optim.step()
                schedule.step(epoch)
                optim.zero_grad()

                # Metrics logging
                logs = compute_metrics(logits, batch['labels'], metrics)
                loss_metric.update(loss)
                logs['loss'] = loss_metric.compute()
                message = [epoch_message] + [
                    f"{k}={v :.{0<abs(v)<2e-4 and '3g' or '4f'}}"
                    for k, v in logs.items()
                ]
                data_and_progress.set_description(" ".join(message))
            data_and_progress.update(1)

        print(f'Epoch {epoch}, starting evaluation...')
        eval_logs, preds = eval_model(model, dev_ds, epoch, metrics)
        save_checkpoint(config.args, model, model_conf=model.config, optim=optim, epoch=epoch,
                        path=os.path.join(config.logdir, 'chkpt.pt'), metadata=config.metadata)
        # Save only the best models by evaluation F1 score
        if best_f1 < eval_logs['f1']:
            print(f'F1 improved from {best_f1} to {eval_logs["f1"]}, saving...')
            save_model(config.args, model, f'{config.args.n}_train_best.pt')
            save_preds(path=os.path.join(config.logdir, 'preds.json'), preds=preds)
            best_f1 = eval_logs['f1']
        history.append(eval_logs)
        config.metadata.data['history'] = history
    return history, model

def save_as_string(obj, path):
    """
    Saves the given object as a JSON string.
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, 'w') as f:
        json.dump(obj, f)

def save_checkpoint(args, model : TokenClassifier, model_conf : TokenClassifierConfig, optim : torch.optim.Optimizer,
                    epoch : int, path : str, metadata  : Metadata = None, best_f1=0):
    """
    Saves model checkpoint during training. Path should include the filename.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
    'optimizer_state_dict': optim.state_dict(),
    'model_state_dict': model.state_dict(),
    'args' : args,
    'config' : model_conf,
    'epoch' : epoch,
    'best_f1' : best_f1
    }, path)

    if metadata is not None: 
        metadata.save(os.path.dirname(path))

def load_from_checkpoint(path, create_model_fn):
    """
    Loads the model checkpoint from the given path. Creates a TokenClassifier instance, 
    with and passes it args from the checkpoint, and flags from the from the arguments.
    The created model loads the state dict from the checkpoint. Also creates an optimizer with 
    a state_dict from the checkpoint. 
    
    Returns a quintuple (model, opitm, epoch, loss, args)
    """
    try:
        chkpt = torch.load(path)
    except RuntimeError:
        chkpt = torch.load(path, map_location='cpu')
    args, epoch = chkpt['args'], chkpt['epoch']
    print(f'Checkpoint args: {args}')
    epoch += 1 # Checkpoints are created after a finished epoch
    print(f'Checkpoint epoch: {epoch}')
    config = chkpt['config']
    print(f'Checkpoint config: {config}')
    model, tokenizer = create_model_fn(args)
    try:
        model.load_state_dict(chkpt['model_state_dict'])
    except RuntimeError:
        compiled_model = torch.compile(model)
        compiled_model.load_state_dict(chkpt['model_state_dict'])
        optim = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay)
        f1 = chkpt['best_f1'] if 'best_f1' in chkpt.keys() else None
        # Fix the checkpoint
        save_checkpoint(args, model, config, optim, epoch - 1, path, best_f1=f1)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay)
    optim.load_state_dict(chkpt['optimizer_state_dict'])
    if not args.lora:
        model.set_base_requires_grad(False)

    if 'best_f1' in chkpt:
        return model, tokenizer, optim, epoch, chkpt['best_f1'], args
    return model, tokenizer, optim, epoch, 0, args


def compute_metrics(y_pred, y, metrics : torchmetrics.MetricCollection):
        """Compute and return metrics given the inputs, predictions, and target outputs."""
        metrics.update(y_pred, y.unsqueeze(-1))
        return metrics.compute()

def create_loss(args):
    # Create a loss function
    if args.focal:
        return partial(sigmoid_focal_loss, alpha=args.pos_weight, reduction='mean')
    
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight]))

def create_model(args):
    # Load ESM-2
    base, tokenizer = get_esm(args)

    loss = create_loss(args)
                                                            
    # Create a classifier
    config = RNNTokenClassiferConfig(n_labels=1, loss=loss, hidden_size=args.hidden_size,
                                     n_layers=args.rnn_layers)
    
    if args.use_cnn:
        config.sr_dim = 256
    if args.lora:
        config.apply_lora = args.lora, 
        config.lora_config=lora.MultiPurposeLoRAConfig(256)
    
    model = RNNTokenClassifier(config, base)

    # Freeze the base if we're not using lora (in that case, it is frozen when applying it)
    if not args.lora:
        model.set_base_requires_grad(False)

    return model, tokenizer

def run_training(args, create_model_fn):
    set_seeds(args.seed)

    log_dirname = args.o if args.o else "{}_{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        )
    # Create logdir name
    args.logdir = os.path.join("logs", log_dirname)
    checkpoint_loaded = False
    if args.checkpoint_path is not None:
        checkpoint_loaded = True
        prev_ft_val = args.fine_tune
        model, tokenizer, optim, epoch, best_f1, args = load_from_checkpoint(args.checkpoint_path, create_model_fn)
    
    
    # Load a model saved with torch.save() 
    elif args.model_path is not None:
        model = load_torch_model(args.model_path)
    else:
        model, tokenizer = create_model_fn(args)


    print(f'batch {args.batch_size} accum {args.accum} effective batch {args.accum * args.batch_size}')
    
    if args.compile:
        # Compile the model, useful in general on Ampere architectures and further
        compiled_model = torch.compile(model)
        compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
        training_model = compiled_model
    else:
        training_model = model.to(device)
    
    # Create metadata
    meta = Metadata()
    meta.data = {'args' : args }
    meta.save(args.logdir)

    train, dev = prepare_datasets(args, tokenizer, model.ignore_index)
    train_config = TrainingConfig(
        epochs = args.epochs, accum=args.accum, batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        lr=args.lr, logdir=args.logdir, metadata=meta,
        args = args
        )
    # --- Training ---
    if checkpoint_loaded:
        train_config.optim = optim
        train_config.f1_min = best_f1
        train_config.start_epoch = epoch
        print('Resuming from checkpoint')

    history, compiled_model = train_model(train_ds=train, dev_ds=dev, model=training_model, config=train_config)

    # --- Fine-tuning ---
    if args.fine_tune:
        
        # Save model before fine-tuning
        save_model(args, model, f'{args.n}_pre_ft')
        meta.data['fine_tuning'] = True
        # Unfreeze base
        model.set_base_requires_grad(True)

        if args.lora:
            model.apply_lora()

        # Recompile model
        if args.compile:
            compiled_model = torch.compile(model)
            compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
            training_model = compiled_model
        else:
            training_model = model.to(device)

        # Train with a lower learning rate
        ft_history, compiled_model = train_model(args, train_ds=train, dev_ds=dev, model=training_model,
                       seed=args.seed, lr=args.lr / 10, metadata=meta)
        history.extend(ft_history)
        meta.data['history'] = history
    save_model(args, model, args.n)
    return history, model

def main(args):
    return run_training(args, create_model)

if __name__ == '__main__':
    args = parser.parse_args()
    history, model = main(args)