import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from model import Transformer

# Use a lower precision for better performance
torch.set_float32_matmul_precision('medium')

class LLaMA(LightningModule):
    def __init__(self,
                 tokenizer: Tokenizer, 
                 config: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.model = Transformer(config)
        self.validation_step_outputs = [] # Used for saving predictions throughout training
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_id)

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        (x, y_true) = batch
        #print('----------------')
        # print('batch_idx: ', batch_idx)
        # for i in range(x.shape[0]):
        #     item = x[i].tolist()
        #     item_2 = y_true[i].tolist()
        #     decoded = self.tokenizer.decode(item)
        #     decoded_2 = self.tokenizer.decode(item_2)
        #     print("DECODED at x: ", i, " ::", decoded)
        #     print("DECODED at y: ", i, " ::", decoded_2)
        #print('training: ', y_true.shape)
        #with autocast(): # autocast is torch package for running in mixed precision, which improves performance
        y_hat = self.model(x)

        #TODO: trying to compute loss only on the response portion, conditioned on the prompt

        loss = self.criterion(y_hat, y_true)

        """if loss < 1:
            sf = torch.nn.Softmax(dim=1)
            y_hat_sf = sf(y_hat)
            # Get highest prob
            y_hat_argmax = torch.argmax(y_hat_sf, dim=1)
            #print('y_true: ', y_true.shape)
            for batch_item in range(len(y_true)):
                #
                #print('y_hat: ', y_hat.shape)
                print('\n')
                print('y_true: ', y_true[batch_item])
                #print('y_hat: ', y_hat[0])
                #print('y_hat_sf: ', y_hat_sf.shape)
                #print('y_hat_sf: ', y_hat_sf[0])
                #print('y_hat_argmax: ', y_hat_argmax.shape)
                print('y_hat_argmax: ', y_hat_argmax[batch_item])

                item = y_hat_argmax[batch_item].tolist()
                item_2 = y_true[batch_item].tolist()
                decoded = self.tokenizer.decode(item)
                decoded_2 = self.tokenizer.decode(item_2)
                print("DECODED at y hat:", decoded)
                print("DECODED at y true:", decoded_2)"""

        loss = loss/self.config.gradient_accumulation_steps

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y_true) = batch
        #print('validation: ', y_true.shape)
        y_hat = self.model(x)
        eval_loss = self.criterion(y_hat, y_true)

        if self.config.save_predictions_during_training:
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(y_hat, 1).detach().cpu().tolist()

            decoded = self.tokenizer.decode(preds)

            self.validation_step_outputs.append(decoded)

        perplexity = torch.exp(eval_loss)
        self.log('val_loss', eval_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return eval_loss

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
