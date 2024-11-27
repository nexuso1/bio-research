import torch
import lightning as L
import torchmetrics
import os.path
import datetime
import matplotlib.pyplot as plt
import io

from torch.utils.data import DataLoader
from data_loading import prep_batch
from functools import partial
from torchvision.transforms import ToTensor
from PIL import Image
from token_classifier_base import TokenClassifier
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils import Metadata
from esm import save_model, device
from lightning.pytorch.loggers import TensorBoardLogger
from data_loading import prepare_datasets
from transformers import AutoTokenizer

class LightningWrapper(L.LightningModule):
    def __init__(self, args, module : TokenClassifier, epoch_metrics : torchmetrics.MetricCollection,
                 step_metrics : torchmetrics.MetricCollection, ds_size : int):
        super(LightningWrapper, self).__init__()
        self.classifier = module
        self.ds_size = ds_size
        self.step_metrics = step_metrics
        self.test_step_metrics = step_metrics.clone(prefix='test_') 
        self.val_step_metrics = step_metrics.clone(prefix='val_')

        self.epoch_metrics = epoch_metrics
        self.test_epoch_metrics = epoch_metrics.clone(prefix='test_')
        self.val_epoch_metrics = epoch_metrics.clone(prefix='val_')

        self.loss_metric = torchmetrics.MeanMetric()
        self.prc = torchmetrics.PrecisionRecallCurve('binary', ignore_index=self.classifier.ignore_index)
        self.test_preds = []
        self.save_hyperparameters(args)

    def _compute_metrics_step(self, logits, labels, step_metrics, epoch_metrics):
        step_metrics.update(logits, labels)
        epoch_metrics.update(logits, labels.int())
        self.prc.update(logits, labels.int())
        self.log_dict(step_metrics.compute(), logger=True, sync_dist=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss, logits = self.classifier.train_predict(**batch)
        self.loss_metric.update(loss)
        self.log('train_loss', self.loss_metric.compute(), logger=True, sync_dist=True, prog_bar=True)
        self._compute_metrics_step(logits.reshape(-1, self.classifier.n_labels), batch['labels'].view(-1, self.classifier.n_labels),
                                   self.step_metrics, self.epoch_metrics)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits = self.classifier.predict(**batch)
        self.loss_metric.update(loss)
        self.log('val_loss', self.loss_metric.compute(), logger=True, prog_bar=True, sync_dist=True)
        self._compute_metrics_step(logits.reshape(-1, self.classifier.n_labels), batch['labels'].view(-1, self.classifier.n_labels), 
                                   self.val_step_metrics, self.val_epoch_metrics)
    
    def test_step(self, batch, batch_idx):
        loss, logits = self.classifier.predict(**batch)
        self.test_preds.append((logits.squeeze(), batch['indices'].squeeze()))
        self.loss_metric.update(loss)
        self.log('test_loss', self.loss_metric.compute(), logger=True, prog_bar=True, sync_dist=True)
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
            torch.save(preds,f'{self.hparams.logdir}/preds.pt')
            torch.save(indices,f'{self.hparams.logdir}/indices.pt')

        self.log_dict(epoch_metrics.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.prc.compute()

        fig, ax = plt.subplots(figsize=(10, 10))

        self.prc.plot(ax=ax, score=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        im = ToTensor()(Image.open(buf))

        self.logger.experiment.add_image(
            "val_prc",
            im,
            global_step=self.current_epoch,
        )

        self.loss_metric.reset()
        epoch_metrics.reset()
        step_metrics.reset()
        self.prc.reset()

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end('val')

    def on_test_epoch_end(self):
        self._shared_epoch_end('test')



    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.classifier.parameters(), 
                                  lr=self.hparams.lr,
                                  weight_decay=self.hparams.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=self.ds_size)
        return {'optimizer' : optim, 'lr_scheduler' : { 
            "scheduler" : schedule,
            "monitor" : "train_loss",
            "frequency" : 1
        }}
    
def train_model(args, train, dev, test, model):
    logger = TensorBoardLogger(args.logdir, name=f'tb_log')
    chkpt_callback = ModelCheckpoint(args.logdir, filename='chkpt', monitor='val_f1')
    es_callback = EarlyStopping('val_loss', patience=40)

    # Use deepspeed 
    if torch.cuda.device_count() > 0:
        strategy = "deepspeed_stage_2"
    else:
        strategy = "auto"
    trainer = L.Trainer(logger=logger, callbacks=[chkpt_callback, es_callback], max_epochs=args.epochs,
                        deterministic=True, log_every_n_steps=1,  accumulate_grad_batches=args.accum, strategy=strategy)
    trainer.fit(model, train, dev)
    test_metrics = trainer.test(model, test)
    print(test_metrics)

    return model, test_metrics

def load_from_checkpoint(checkpoint_path, create_model_fn):
    chkpt = torch.load(checkpoint_path)
    model = create_model_fn(chkpt['hyper_parameters'])
    return LightningWrapper.load_from_checkpoint(chkpt, module=model)

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
    if args.checkpoint_path is not None:
        model, tokenizer, args = load_from_checkpoint(args.checkpoint_path, create_model_fn)
    else:
        model, tokenizer = create_model_fn(args)

    return model, tokenizer

def run_training(args, create_model_fn):
    L.seed_everything(args.seed)

    log_dirname = args.o if args.o else "{}_{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        )

    args.logdir = os.path.join("logs", log_dirname)
    
    tokenizer = get_tokenizer(args)
    
    full_dataset = prepare_datasets(args, tokenizer, ignore_label=args.ignore_label)

    step_metrics = torchmetrics.MetricCollection({
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=args.ignore_label),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=args.ignore_label),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=args.ignore_label),
    })

    epoch_metrics = torchmetrics.MetricCollection({
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=args.ignore_label),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=args.ignore_label),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=args.ignore_label),
        'auroc' : torchmetrics.AUROC('binary', ignore_index=args.ignore_label),
        'auprc' : torchmetrics.AveragePrecision('binary', ignore_index=args.ignore_label),
        'mcc' : torchmetrics.MatthewsCorrCoef('binary', ignore_index=args.ignore_label)
    })

    # Create metadata
    meta = Metadata()
    meta.data = {'args' : args }
    meta.save(args.logdir)
    test = DataLoader(full_dataset.test_ds, args.batch_size, shuffle=False, 
                      collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label),
                      persistent_workers=True if args.num_workers > 0 else False,
                      num_workers=args.num_workers)
    
    master_logdir = args.logdir
    metric_hist = {}
    for i in range(full_dataset.n_splits):
        args.logdir = os.path.join(master_logdir, f'fold_{i}')
        train_ds, dev_ds = full_dataset.get_fold(i)
        
        train = DataLoader(train_ds, args.batch_size, shuffle=True,
                            collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label),
                            persistent_workers=True if args.num_workers > 0 else False, 
                            num_workers=args.num_workers )
        dev = DataLoader(dev_ds, args.batch_size, shuffle=False,
                            collate_fn=partial(prep_batch, tokenizer=tokenizer, ignore_label=args.ignore_label),
                            persistent_workers=True if args.num_workers > 0 else False,
                            num_workers=args.num_workers)
        
        model, tokenizer = prepare_model(args, create_model_fn)
        model = LightningWrapper(args, model, step_metrics=step_metrics, epoch_metrics=epoch_metrics, ds_size=len(train))

        if args.compile:
            # Compile the model, useful in general on Ampere architectures and further
            compiled_model = torch.compile(model)
            compiled_model.to(device) # We cannot save the compiled model, but it shares weights with the original, so we save that instead
            training_model = compiled_model
        else:
            training_model = model.to(device)
    
        training_model, test_metrics = train_model(args, train, dev, test, training_model)
        monitor_metric = 'test_f1'
        monitor_best_val = 0

        if len(metric_hist.keys()) == 0:
            metric_hist = test_metrics
        
        print(f'Test metric averages after epoch {i}')
        for metric in metric_hist[0]:
            metric_hist[metric] += test_metrics[metric]
            print(f'{metric}: {metric_hist[metric_hist] / i}')

        if monitor_best_val > test_metrics[monitor_metric]:
            monitor_best_val = test_metrics[monitor_metric]
            save_model(args, model, args.n)
    return model
