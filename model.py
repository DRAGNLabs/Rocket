import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pathlib import Path
from transformers import (
    LlamaForCausalLM as LanguageModel, 
    LlamaConfig as HFConfig
)

from tokenizer.tokenizer import Tokenizer

# Use a lower precision for better performance
torch.set_float32_matmul_precision('medium')

class Model(LightningModule):
    def __init__(self,
                 tokenizer: Tokenizer, 
                 config: dict = None):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        if config.from_pretrained is not True:
            # * Configure necessary HF model parameters here
            model_config = HFConfig(
                vocab_size = config.vocab_size,
                max_position_embeddings = config.max_sequence_embeddings,
                hidden_size=config.dim,
                num_hidden_layers=config.n_layers,
                num_attention_heads=config.n_heads,
                rms_norm_eps=config.norm_eps,
                pad_token_id=config.pad_id
            )
            self.model = LanguageModel(model_config)
        elif config.from_pretrained is True and config.model_name is not None:
            self.model = LanguageModel.from_pretrained(config.model_name)
        else:
            raise ValueError("Must provide model_name if from_pretrained is True")
        
        self.validation_step_outputs = [] # Used for saving predictions throughout training
        #self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_id)

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        x, x_mask, y_true = batch

        #with autocast(): # autocast is torch package for running in mixed precision, which improves performance
        #y_hat = self.model(x)
        output = self.model(input_ids=x, attention_mask=x_mask, labels=y_true)
        #TODO: trying to compute loss only on the response portion, conditioned on the prompt

        #loss = self.criterion(y_hat, y_true)
        loss = output.loss

        loss = loss/self.config.gradient_accumulation_steps

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_mask, y_true = batch

        #y_hat = self.model(x)
        #val_loss = self.criterion(y_hat, y_true)
        output = self.model(input_ids=x, attention_mask=x_mask, labels=y_true)
        val_loss = output.loss
        y_hat = output.logits
        #print('loss: ', val_loss)

        if self.config.save_predictions_during_training:
            # Decode predictions and add to valuation predictions list
            #print('logits shape: ', y_hat.shape) # 32, 1024, 10000
            probs = torch.softmax(y_hat, dim=2)
            #print('probs shape: ', probs.shape) # 32, 1024, 10000
            preds = torch.argmax(probs, 2).detach().cpu().tolist()
            #print('y_true preds: ', y_true[0])
            #print('preds: ', preds[0]) # 32, 1024
            #print('mask: ', x_mask[0])

            y_true_decoded = self.tokenizer.decode(y_true[0].tolist())
            decoded = self.tokenizer.decode(preds[0])
            #print('y_true_decoded: ', y_true_decoded)
            #print('decoded: ', decoded)
            #y_true_decoded = self.tokenizer.decode(y_true[0].tolist())

            self.validation_step_outputs.append(decoded)

        perplexity = torch.exp(val_loss)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        return val_loss
    
    # TODO: add test function

    def on_validation_epoch_end(self) -> None:
        if self.config.save_predictions_during_training == True:
            dir_path = Path(self.config.default_root_dir)
            file_path = dir_path / 'validation_predictions.txt'

            # Check if the directory exists. If not, create it
            dir_path.mkdir(parents=True, exist_ok=True)

            # Check if the file exists. If not, create it and append the outputs
            with file_path.open('a', encoding="utf-8") as f:
                for item in self.validation_step_outputs:
                    f.write(str(self.current_epoch) + ': ')
                    f.write(str(item) + '\n')

            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)  # model.paramaters = weights tensor
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer]#, [lr_scheduler]
