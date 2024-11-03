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
from tqdm.auto import tqdm
from utils import Metadata, load_torch_model
from data_loading import prepare_datasets
from datasets import Dataset
from transformers import set_seed, EsmModel, AutoTokenizer
from token_classifier_base import TokenClassifier, TokenClassifierConfig
from classifiers import RNNTokenClassifier, RNNTokenClassiferConfig
from torchvision.ops import sigmoid_focal_loss


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size)', default=4)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=20)
parser.add_argument('--max_length', type=int, help='Maximum sequence length (shorter sequences will be pruned)', default=1024)
parser.add_argument('--prot_info_path', type=str, 
                     help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.',
                     default='../data/phosphosite_sequences/phosphosite_df_small.json')
parser.add_argument('--train_path', type=str, help='Path to train protein IDs, subset of IDs in the prot. info dataset. JSON list.',
                    default='../data/cleaned_train_prots.json')
parser.add_argument('--test_path', type=str, help='Path to test protein IDs, subset of IDs in the prot. info dataset. JSON list.',
                    default='../data/cleaned_test_prots.json')
parser.add_argument('--fine_tune', action='store_true', help='Use fine tuning on the base model or not. Default is False', default=False)
parser.add_argument('--ft_only', action='store_true', help='Skip pre-training, only fine-tune', default=False)
parser.add_argument('--weight_decay', type=float, help='Weight decay', default=0.01)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=3)
parser.add_argument('--val_batch', type=int, help='Validation batch size', default=10)
parser.add_argument('--hidden_size', type=int, help='Classifier hidden size', default=256)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
parser.add_argument('-o', type=str, help='Output folder', default=None)
parser.add_argument('-n', type=str, help='Model name', default='esm')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model')
parser.add_argument('--lora', action='store_true', help='Use LoRA', default=False)
parser.add_argument('--dropout', type=float, help='Dropout probability', default=0)
parser.add_argument('--ft_epochs', type=int, help='Number of epochs for finetuning', default=10)
parser.add_argument('--type', help='ESM Model type', type=str, default='650M')
parser.add_argument('--pos_weight', help='Positive class weight', type=float, default=0.75)
parser.add_argument('--num_workers', help='Number of multiprocessign workers', type=int, default=0)
parser.add_argument('--rnn_layers', help='Number of RNN classifier layers', type=int, default=2)
parser.add_argument('--checkpoint_path', help='Resume training from checkpoint', type=str, default=None)
parser.add_argument('--model_path', help='Load model from this path (not a checkpoint)', type=str, default=None)
parser.add_argument('--use_cnn', help='Use CNN seq reps', action='store_true', default=False)
parser.add_argument('--focal', help='Use focal loss. In this mode, pos_weight will be treated as the alpha parameter.', action='store_true', default=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
            valid_logits = logits[batch['labels'] != -1] # Gather valid logits
            indices = np.cumsum(batch['batch_lens'].cpu().numpy(), 0) # Indices into gathered logits according to batch lengths
            probs = np.split(valid_logits.cpu().numpy(), indices)[:-1] # Split them according to the batch lenghts, last element is extra
            probs_list.extend(probs) # Predicted probabilites
    return {k : v.cpu().numpy() for k, v in logs.items()}, probs_list

def train_model(args, train_ds : Dataset, dev_ds : Dataset, model : TokenClassifier, lr, metadata : Metadata=None, seed=42,
                start_epoch=0, f1_min=0, optim=None):

    # Set all random seeds
    set_seeds(seed)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay) if optim is None else optim

    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, len(train_ds) * args.epochs)

    metrics = {
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=model.ignore_index).to(device),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=model.ignore_index).to(device),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=model.ignore_index).to(device),
    }
    loss_metric =  torchmetrics.MeanMetric().to(device)
    metrics = torchmetrics.MetricCollection(metrics)

    history = []
    best_f1 = f1_min
    # Train model
    for epoch in range(start_epoch, args.epochs):
        model.train()
        metrics.reset()
        epoch_message = f"Epoch={epoch+1}/{args.epochs}"
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
            loss = loss / args.accum

            if args.accum == 1 or ( i > 0 and i % args.accum == 0) or (i + 1 == len(train_ds)):
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
        save_checkpoint(args, model, config=model.config, optim=optim, epoch=epoch,
                        path=os.path.join(args.logdir, 'chkpt.pt'), metadata=metadata)
        # Save only the best models by evaluation F1 score
        if best_f1 < eval_logs['f1']:
            print(f'F1 improved from {best_f1} to {eval_logs["f1"]}, saving...')
            save_model(args, model, f'{args.n}_train_best.pt')
            save_preds(path=os.path.join(args.logdir, 'preds.json'), preds=preds)
            best_f1 = eval_logs['f1']
        history.append(eval_logs)
        metadata.data['history'] = history
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

def save_checkpoint(args, model : TokenClassifier, config : TokenClassifierConfig, optim : torch.optim.Optimizer,
                    epoch : int, path : str, metadata  : Metadata = None, best_f1=0):
    """
    Saves model checkpoint during training. Path should include the filename.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
    'optimizer_state_dict': optim.state_dict(),
    'model_state_dict': model.state_dict(),
    'args' : args,
    'config' : config,
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
    chkpt = torch.load(path)
    args, epoch = chkpt['args'], chkpt['epoch']
    print(f'Checkpoint args: {args}')
    epoch += 1 # Checkpoints are created after a finished epoch
    print(f'Checkpoint epoch: {epoch}')
    config = chkpt['config']
    print(f'Checkpoint config: {config}')
    model, tokenizer = create_model_fn(args)
    model.load_state_dict(chkpt['model_state_dict'])
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(),weight_decay=args.weight_decay)
    optim.load_state_dict(chkpt['optimizer_state_dict'])
    if not args.lora:
        model.set_base_requires_grad(False)

    if 'best_f1' in chkpt:
        return model, tokenizer, optim, epoch, chkpt['best_f1'], args
    return model, tokenizer, optim, epoch, 0, args

def save_model(args, model : TokenClassifier, name : str):
    """
    Saves the model to the folder args.o if given, otherwise to args.logdir, with the given name.
    """
    if args.o is None:
        folder = args.logdir
    else:
        folder = args.o

    save_path = f'{folder}/{name}.pt'
    if not os.path.exists(f'{folder}'):
        os.mkdir(f'{folder}')

    model.save(save_path)
    print(f'Model saved to {save_path}')

def compute_metrics(y_pred, y, metrics : torchmetrics.MetricCollection):
        """Compute and return metrics given the inputs, predictions, and target outputs."""
        metrics.update(y_pred, y.unsqueeze(-1))
        return metrics.compute()

def create_loss(args):
    # Create a loss function
    if args.focal:
        return partial(sigmoid_focal_loss, alpha=args.pos_weight)
    
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

    # Create logdir name
    args.logdir = os.path.join(
        "logs",
        "{}_{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        ),
    )
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

    # --- Training ---
    if checkpoint_loaded:
        print('Resuming from checkpoint...')
        history, compiled_model = train_model(args, train_ds=train, dev_ds=dev, model=training_model, seed=args.seed, lr=args.lr,
                                              optim=optim, start_epoch=epoch, metadata=meta, f1_min=best_f1)
    elif not args.fine_tune or args.fine_tune and not args.ft_only:
        history, compiled_model = train_model(args, train_ds=train, dev_ds=dev, model=training_model, seed=args.seed, lr=args.lr, metadata=meta)

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