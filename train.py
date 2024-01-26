import sys
import signal
import yaml

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger

from dataset import DataModule
from tokenizer.tokenizer import Tokenizer
from model import Model
from utils.data_utils import Struct

torch.set_float32_matmul_precision('medium')

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training started")
    def on_train_end(self, trainer, pl_module):
        print("Training ended")

def train(config):
    seed_everything(config.seed, workers=True)
    tokenizer = Tokenizer(model_path=config.tokenizer_path)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    # Build model class
    model = Model(tokenizer=tokenizer, 
                  config=config)

    dm = DataModule(config.train_path, 
                    config.val_path, 
                    tokenizer, 
                    config.batch_size, 
                    config.max_sequence_embeddings)

    # callbacks
    early_stopping = EarlyStopping('val_loss', patience=config.early_stopping, mode='min', verbose=True)
    logger = CSVLogger(save_dir=config.default_root_dir, name='logs')
    model_checkpoint = ModelCheckpoint(
        dirpath=config.default_root_dir + '/checkpoints',
        filename='model-{epoch}-{val_loss:.2f}',
        save_top_k=config.save_top_k,
        monitor='val_loss',
        mode='min')
    print_callback = PrintCallback()

    # Train
    if not config.use_slurm:
        trainer = Trainer(
            default_root_dir=config.default_root_dir,
            accelerator=config.accelerator,
            val_check_interval=config.val_check_interval,
            log_every_n_steps=config.log_every_n_steps,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            max_epochs=config.num_epochs,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            sync_batchnorm=True,
            callbacks=[early_stopping, print_callback, model_checkpoint],
            logger=logger
            )
    else:
        trainer = Trainer(
            default_root_dir=config.default_root_dir,
            accelerator=config.accelerator,
            val_check_interval=config.val_check_interval,
            log_every_n_steps=config.log_every_n_steps,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            num_nodes=config.num_nodes,
            devices=config.devices,
            strategy="ddp",
            max_epochs=config.num_epochs,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            sync_batchnorm=True,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
            callbacks=[early_stopping, print_callback, model_checkpoint],
            logger=logger
            )
        
    trainer.fit(model, datamodule=dm)

    print('\nNo errors!\n')

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    train(config)

if __name__ == "__main__":
    main()