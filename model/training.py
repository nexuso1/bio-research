# This source file is the backbone of model training. When training
# a model, we use the run_training function defined in this 
# file

import torch
import lightning as L
import os.path
import datetime
import matplotlib.pyplot as plt
import io
import json

from torchmetrics import F1Score, MatthewsCorrCoef, Precision, Recall, AUROC, \
MeanMetric, AveragePrecision, PrecisionRecallCurve, MetricCollection
from torch.utils.data import DataLoader
from data_loading import prep_batch
from functools import partial
from torchvision.transforms import ToTensor
from PIL import Image
from token_classifier_base import TokenClassifier
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils import Metadata, sigmoid_focal_loss
from lightning.pytorch.loggers import TensorBoardLogger
from data_loading import prepare_datasets
from transformers import AutoTokenizer
from pathlib import Path
from argparse import Namespace, ArgumentParser

parser = ArgumentParser()

parser.add_argument('--seed', type=int, help='Random seed', default=42)
parser.add_argument('--batch_size', type=int, help='Batch size)', default=4)
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=20)
parser.add_argument('--prot_info_path', type=str, 
                     help='Path to the protein dataset. Expects a dataframe with columns ("id", "sequence", "sites"). "sequence" is the protein AA string, "sites" is a list of phosphorylation sites.',
                     default='../data/phosphosite_sequences/phosphosite_df.json')
parser.add_argument('--dataset_path', type=str, help='Path to the prepared dataset, with information about train, test sets; and folds.', default='../data/splits_S.json')
parser.add_argument('--weight_decay', type=float, help='Weight decay', default=1e-4)
parser.add_argument('--accum', type=int, help='Number of gradient accumulation steps', default=3)
parser.add_argument('--hidden_size', type=int, help='Classifier hidden size', default=128)
parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
parser.add_argument('-o', type=str, help='Output folder', default=None)
parser.add_argument('-n', type=str, help='Model name', default='esm')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model')
parser.add_argument('--lora', action='store_true', help='Use LoRA', default=False)
parser.add_argument('--dropout', type=float, help='Dropout probability', default=0)
parser.add_argument('--type', help='ESM Model type', type=str, default='650M')
parser.add_argument('--pos_weight', help='Positive class weight', type=float, default=3)
parser.add_argument('--num_workers', help='Number of multiprocessign workers', type=int, default=0)
parser.add_argument('--n_layers', help='Number of RNN/Transformer classifier layers', type=int, default=1)
parser.add_argument('--checkpoint_path', help='Resume training from checkpoint', type=str, default=None)
parser.add_argument('--model_path', help='Load model from this path (not a checkpoint)', type=str, default=None)
parser.add_argument('--focal', help='Use focal loss. In this mode, pos_weight will be treated as the alpha parameter.', action='store_true', default=False)
parser.add_argument('--residues', help='List of residues to train on', default="['S', 'T', 'Y']", type=str)
parser.add_argument('--ignore_label', help='Label that will be ignored by the loss', default=-1, type=int)
parser.add_argument('--patience', help='Patience during training', default=20, type=int)
parser.add_argument('--debug', help='Debug mode', default=False, action='store_true')
parser.add_argument('--step_lr', help='Use StepLR scheduler', default=False, action='store_true')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LightningWrapper(L.LightningModule):
    def __init__(self, args, module : TokenClassifier, epoch_metrics : MetricCollection,
                 step_metrics : MetricCollection, ds_size : int, logdir : str):
        super(LightningWrapper, self).__init__()
        self.classifier = module
        self.ds_size = ds_size
        self.step_metrics = step_metrics
        self.logdir = logdir
        self.test_step_metrics = step_metrics.clone(prefix='test_') 
        self.val_step_metrics = step_metrics.clone(prefix='val_')

        self.epoch_metrics = epoch_metrics
        self.test_epoch_metrics = epoch_metrics.clone(prefix='test_')
        self.val_epoch_metrics = epoch_metrics.clone(prefix='val_')

        self.loss_metric = MeanMetric()
        self.prc = PrecisionRecallCurve('binary', ignore_index=self.classifier.ignore_index)
        self.test_preds = []
        self.debug = False
        if hasattr(args, "debug") and args.debug:
            self.debug = True
            self.print_counter = 0
            
        self.save_hyperparameters(args)

    def _compute_metrics_step(self, logits, labels, step_metrics, epoch_metrics):
        step_vals = step_metrics(logits, labels)
        epoch_metrics.update(logits, labels.int())
        self.prc.update(logits, labels.int())
        self.log_dict(step_vals, sync_dist=True, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        loss, logits = self.classifier.train_predict(**batch)
        mean_loss = self.loss_metric(loss)
        self.log('train_loss', loss, logger=True, prog_bar=True, sync_dist=True)
        self._compute_metrics_step(logits.reshape(-1, self.classifier.n_labels), batch['labels'].view(-1, self.classifier.n_labels),
                                   self.step_metrics, self.epoch_metrics)
        
        if self.debug:
            if self.print_counter >= 250:
                with torch.no_grad():
                    print(batch)
                    print(torch.sigmoid(logits))
                self.print_counter = 0
            
            self.print_counter += 1

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits = self.classifier.predict(**batch)
        mean_loss = self.loss_metric(loss)
        self.log('val_loss', loss, logger=True, prog_bar=True, sync_dist=True)
        self._compute_metrics_step(logits.reshape(-1, self.classifier.n_labels), batch['labels'].view(-1, self.classifier.n_labels), 
                                   self.val_step_metrics, self.val_epoch_metrics)
    
    def test_step(self, batch, batch_idx):
        loss, logits = self.classifier.predict(**batch)
        self.test_preds.append((logits.squeeze(), batch['indices'].squeeze()))
        mean_loss = self.loss_metric(loss)
        self.log('test_loss', loss, logger=True, prog_bar=True, sync_dist=True)
        self.log('test_loss_mean', mean_loss, logger=True, prog_bar=True, sync_dist=True)
        self._compute_metrics_step(logits.reshape(-1, self.classifier.n_labels), batch['labels'].view(-1, self.classifier.n_labels), 
                                   self.test_step_metrics, self.test_epoch_metrics)

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.epoch_metrics.compute(), logger=True, prog_bar=True, sync_dist=True)
        self.loss_metric.reset()
        self.epoch_metrics.reset()
        self.step_metrics.reset() 
        self.prc.reset()
 
    def _shared_epoch_end(self, mode='val'):
        if mode == 'val':
            epoch_metrics = self.val_epoch_metrics
            step_metrics = self.val_step_metrics
        else:
            epoch_metrics = self.test_epoch_metrics
            step_metrics = self.test_step_metrics

        if mode == 'test':
            preds, indices = zip(*self.test_preds)
            torch.save(preds,f'{self.logdir}/preds.pt')
            torch.save(indices,f'{self.logdir}/indices.pt')
        
        self.log_dict(epoch_metrics.compute(), prog_bar=True, logger=True, sync_dist=True)

        fig, ax = plt.subplots(figsize=(10, 10))

        self.prc.plot(ax=ax, score=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        im = ToTensor()(Image.open(buf))

        self.logger.experiment.add_image(
            f"{mode}_prc",
            im,
            global_step=self.current_epoch,
        )

        self.loss_metric.reset()
        epoch_metrics.reset()
        step_metrics.reset()
        self.prc.reset()

        plt.close('all')

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end('val')

    def on_test_epoch_end(self):
        self._shared_epoch_end('test')

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.classifier.parameters(), 
                                  lr=self.hparams.lr,
                                  betas=(0.9, 0.98),
                                  weight_decay=self.hparams.weight_decay)
        if self.hparams.step_lr:
            # Needed for UniPTM training
            schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.92)
        else:
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.hparams.epochs)
        return {'optimizer' : optim, 'lr_scheduler' : { 
            "scheduler" : schedule,
            "interval": "epoch",
            "monitor" : "train_loss",
            "frequency" : 1
        }}

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

def create_loss(args):
    # Create a loss function
    if args.focal:
        return partial(sigmoid_focal_loss, alpha=args.pos_weight, reduction='mean')
    
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos_weight]))
    
def train_model(args, train, dev, test, model : LightningWrapper, logdir):
    logger = TensorBoardLogger(logdir, name=f'tb_log')

    # Best model checkpoint
    best_callback = ModelCheckpoint(logdir, filename='best', monitor='val_f1', mode='max',
                                      save_on_train_epoch_end=1, auto_insert_metric_name=True)
    # Training checkpoint (because having a defined ModelCheckpoint overrides the default checkpointing)
    chkpt_callback = ModelCheckpoint(logdir, filename='chkpt')
    es_callback = EarlyStopping('val_f1', patience=args.patience, mode="max")

    # Use deepspeed 
    if torch.cuda.device_count() > 1:
        strategy = "deepspeed_stage_2"
    else:
        strategy = "auto"
    trainer = L.Trainer(logger=logger, callbacks=[best_callback, es_callback, chkpt_callback], max_epochs=args.epochs,
                        deterministic=True, log_every_n_steps=1,  accumulate_grad_batches=args.accum, strategy=strategy,
                        default_root_dir=logdir)
    trainer.fit(model, train, dev, ckpt_path=args.checkpoint_path)
    best = torch.load(f'{logdir}/best.ckpt')
    model.load_state_dict(best['state_dict'])
    test_metrics = trainer.test(model, test)
    print(test_metrics)

    return model, test_metrics

def get_tokenizer(args):
    if args.type == '3B' :
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t36_3B_UR50D')
    elif type == '15B':
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')
    elif type == '35M':
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    else:
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

    return tokenizer

def prepare_model(args, create_model_fn):
    model, tokenizer = create_model_fn(args)

    return model, tokenizer

def run_training(args : Namespace, create_model_fn):
    L.seed_everything(args.seed)

    log_dirname = args.o if args.o else "{}_{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        )

    args.logdir = os.path.join("new_logs", log_dirname)

    if not args.checkpoint_path:
        # Create metadata
        meta = Metadata()
        meta.data = {'args' : args }
        meta.data['current_fold'] = 0
        meta.data['test_metrics'] = []
        meta.save(args.logdir)
    else:
        par_dir = Path(args.checkpoint_path).parent
        chkpt_path = args.checkpoint_path
        with open(f'{par_dir.parent}/metadata.json', 'r') as f:
            meta = Metadata(**json.load(f))
            if 'current_fold' not in meta.data:
                meta.data['current_fold'] = int(par_dir.name[-1])
            for k, v in meta.data['args'].items():
                args.__setattr__(k, v)
        args.checkpoint_path = chkpt_path
    
    tokenizer = get_tokenizer(args)
    
    full_dataset = prepare_datasets(args, ignore_label=args.ignore_label)

    step_metrics = MetricCollection({
        'f1' : F1Score(task='binary', ignore_index=args.ignore_label),
        'precision' : Precision(task='binary',ignore_index=args.ignore_label),
        'recall' : Recall(task='binary', ignore_index=args.ignore_label),
    })

    epoch_metrics = MetricCollection({
        'f1' : F1Score(task='binary', ignore_index=args.ignore_label),
        'precision' : Precision(task='binary',ignore_index=args.ignore_label),
        'recall' : Recall(task='binary', ignore_index=args.ignore_label),
        'auroc' : AUROC('binary', ignore_index=args.ignore_label),
        'auprc' : AveragePrecision('binary', ignore_index=args.ignore_label),
        'mcc' : MatthewsCorrCoef('binary', ignore_index=args.ignore_label)
    })

    master_logdir = args.logdir
    for i in range(meta.data['current_fold'], full_dataset.n_splits):
        meta.data['current_fold'] = i
        meta.save(master_logdir)
        print(f'Current fold: {i}')
        train_ds, dev_ds, test_ds = full_dataset.get_fold(i)
        
        train = DataLoader(train_ds, args.batch_size, shuffle=True,
                            collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label),
                            persistent_workers=True if args.num_workers > 0 else False, 
                            num_workers=args.num_workers )
        dev = DataLoader(dev_ds, args.batch_size, shuffle=False,
                            collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label),
                            persistent_workers=True if args.num_workers > 0 else False,
                            num_workers=args.num_workers)
        
        test = DataLoader(test_ds, args.batch_size, shuffle=False,
                            collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label),
                            persistent_workers=True if args.num_workers > 0 else False,
                            num_workers=args.num_workers)
        
        model, tokenizer = prepare_model(args, create_model_fn)

        logdir = os.path.join(master_logdir, f'fold_{i}')
        if not isinstance(model, LightningWrapper):
            model = LightningWrapper(args, model, step_metrics=step_metrics, epoch_metrics=epoch_metrics, ds_size=len(train), logdir=logdir)

        if args.compile:
            # Compile the model, useful in general on Ampere architectures and further
            compiled_model = torch.compile(model)
            compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
            training_model = compiled_model
        else:
            training_model = model.to(device)
    
        training_model, test_metrics = train_model(args, train, dev, test, training_model, logdir)
        meta.data['test_metrics'].append(test_metrics[0])

        print(f'Test metrics for fold {i}')
        print(meta.data['test_metrics'][i])
        # Handled by checkpoints
        #save_model(args, model, args.n)

        if args.checkpoint_path:
            # Clear the checkpoint after resuming
            args.checkpoint_path = None
        meta.save(master_logdir)

    print('Overall test metric averages')
    buffer = {k : 0 for k in meta.data['test_metrics'][-1].keys()}
    for i in range(len(meta.data['test_metrics'])):
        fold_metrics = meta.data['test_metrics'][i]
        for k, v in fold_metrics.items():
            buffer[k] += v

    for k, v in buffer.items():
        buffer[k] = v / len(meta.data['test_metrics'])
        print(f'mean {k} : {buffer[k]}')

    meta.data['test_metric_avg'] = buffer
    meta.save(master_logdir)
    return model
