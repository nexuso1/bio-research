import torch
import lightning as L
import torchmetrics
import os.path
import datetime

from token_classifier_base import TokenClassifier
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import Metadata
from esm import save_model, device
from lightning.pytorch.loggers import TensorBoardLogger
from data_loading import prepare_datasets

class LightningWrapper(L.LightningModule):
    def __init__(self, args, module : TokenClassifier, metrics : torchmetrics.MetricCollection):
        super(LightningWrapper, self).__init__()
        self.classifier = module
        self.train_metrics = metrics
        self.valid_metrics = metrics.clone(prefix='val_')

        self.save_hyperparameters(args)

    def training_step(self, batch, batch_idx):
        loss, logits = self.classifier.train_predict(**batch)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        self.train_metrics(logits.view(-1, self.classifier.n_labels), batch['labels'].view(-1, self.classifier.n_labels))
        self.log_dict(self.train_metrics, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, logits = self.classifier.train_predict(**batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.valid_metrics(logits.view(-1, self.classifier.n_labels), batch['labels'].view(-1, self.classifier.n_labels))
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
	
        optim = torch.optim.AdamW(self.classifier.parameters(), 
                                  lr=self.hparams.lr,
                                  weight_decay=self.hparams.weight_decay)
        
        return optim
    
def train_model(args, train, dev, model):
    logger = TensorBoardLogger(args.logdir, name=f'tb_log')
    chkpt_callback = ModelCheckpoint(args.o, filename='chkpt.pt', monitor='val_f1')
    trainer = L.Trainer(logger=logger, callbacks=[chkpt_callback], max_epochs=args.epochs,
                        deterministic=True, log_every_n_steps=1, )
    trainer.fit(model, train, dev, accumulate_grad_batches=args.accum)

def load_from_checkpoint(checkpoint_path, create_model_fn):
    chkpt = torch.load(checkpoint_path)
    model = create_model_fn(chkpt['hyper_parameters'])
    return LightningWrapper.load_from_checkpoint(chkpt, module=model)

def run_training(args, create_model_fn):
    L.seed_everything(args.seed)

    log_dirname = args.o if args.o else "{}_{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        )

    args.logdir = os.path.join("logs", log_dirname)
    checkpoint_loaded = False
    if args.checkpoint_path is not None:
        model, tokenizer, optim, epoch, best_f1, args = load_from_checkpoint(args.checkpoint_path, create_model_fn)
    else:
        model, tokenizer = create_model_fn(args)
    
    train, dev = prepare_datasets(args, tokenizer, ignore_label=model.ignore_index)

    metrics = torchmetrics.MetricCollection({
        'f1' : torchmetrics.F1Score(task='binary', ignore_index=model.ignore_index),
        'precision' : torchmetrics.Precision(task='binary',ignore_index=model.ignore_index),
        'recall' : torchmetrics.Recall(task='binary', ignore_index=model.ignore_index),
        'auroc' : torchmetrics.AUROC('binary', ignore_index=model.ignore_index),
        'mcc' : torchmetrics.MatthewsCorrCoef('binary', ignore_index=model.ignore_index)
    })

    model = LightningWrapper(args, model, metrics=metrics)
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


    training_model = train_model(args, train, dev, training_model)
    save_model(args, model, args.n)
    return model
