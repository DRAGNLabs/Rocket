import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from transformers import (
    LlamaForCausalLM as LanguageModel, 
    LlamaConfig as HFConfig
)

# Use a lower precision for better performance
torch.set_float32_matmul_precision('medium')

class Model(LightningModule):
    def __init__(self,
                 tokenizer: Tokenizer, 
                 config: dict = None,
                 model_name: str = None):
        super().__init__()
        if config is not None:
            self.model = LanguageModel(config)
            self.config = config
        elif model_name is not None:
            self.model = LanguageModel.from_pretrained(model_name)
            self.config = self.model.config
        else:
            raise ValueError("Must provide either config or model_name")
        self.tokenizer = tokenizer
        self.validation_step_outputs = [] # Used for saving predictions throughout training
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_id)

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        (x, y_true) = batch

        #with autocast(): # autocast is torch package for running in mixed precision, which improves performance
        y_hat = self.model(x)

        #TODO: trying to compute loss only on the response portion, conditioned on the prompt

        loss = self.criterion(y_hat, y_true)

        loss = loss/self.config.gradient_accumulation_steps

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y_true) = batch
        #print('validation: ', y_true.shape)
        y_hat = self.model(x)
        val_loss = self.criterion(y_hat, y_true)
        #print('val_loss: ', val_loss)

        if self.config.save_predictions_during_training:
            # Decode predictions and add to valuation predictions list
            preds = torch.argmax(y_hat, 1).detach().cpu().tolist()

            decoded = self.tokenizer.decode(preds)
            #print('decoded: ', decoded)
            #y_true_decoded = self.tokenizer.decode(y_true[0].tolist())
            #print('y_true_decoded: ', y_true_decoded)

            self.validation_step_outputs.append(decoded)

        perplexity = torch.exp(val_loss)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return val_loss

    def on_validation_epoch_end(self) -> None:
        if self.config.save_predictions_during_training == True:
            dir_path = Path(self.config.default_root_dir)
            file_path = dir_path / 'validation_predictions.txt'

            # Check if the directory exists. If not, create it
            dir_path.mkdir(parents=True, exist_ok=True)

            # Check if the file exists. If not, create it and append the outputs
            with file_path.open('a') as f:
                for item in self.validation_step_outputs:
                    f.write(str(self.current_epoch) + ': ')
                    f.write(str(item) + '\n')

            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)  # model.paramaters = weights tensor
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]
