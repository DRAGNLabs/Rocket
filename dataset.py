import torch
import os
import pandas as pd
from typing import List, Optional
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
#import dask.dataframe as dd

class DataModule(LightningDataModule):
    def __init__(self, train_path, val_path, tokenizer, batch_size, sequence_length, num_workers=0):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = DataSet(self.train_path, 
                                            pad_tok=self.tokenizer.pad_id, 
                                            bos_tok=self.tokenizer.bos_id, 
                                            eos_tok=self.tokenizer.eos_id, 
                                            sequence_length=self.sequence_length)
        self.val_dataset = DataSet(self.val_path, 
                                            pad_tok=self.tokenizer.pad_id, 
                                            bos_tok=self.tokenizer.bos_id, 
                                            eos_tok=self.tokenizer.eos_id, 
                                            sequence_length=self.sequence_length)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

class DataSet(torch.utils.data.Dataset):
    def __init__(self, path_to_data, pad_tok, bos_tok, eos_tok, sequence_length):
        assert os.path.isfile(path_to_data), path_to_data
        self.data:pd.DataFrame = pd.read_pickle(path_to_data) # TODO: lazy load this?
        #self.data = dd.read_pickle(self.path_to_data)
        
        self.pad_tok = pad_tok
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.sequence_length = sequence_length

    def __len__(self):
        #print(len(self.data))
        return len(self.data)
    
    def __getitem__(self, index):
        pd_series_item = self.data.iloc[index,:]  # Returns a pd.Series
        tensor_item:List[int] = pd_series_item.iloc[0]  # Grab text from series

        # Handle Padding
        if len(tensor_item) <= self.sequence_length:
            n:int = self.sequence_length - len(tensor_item)
            pads:List[int] = ([self.pad_tok]*(n+1))
            tensor_item:List[int] = tensor_item + pads

            if len(tensor_item) != self.sequence_length+1:
                tensor_item = tensor_item[:self.sequence_length]

        x = torch.tensor(tensor_item[:self.sequence_length])
        y_true = torch.tensor(tensor_item[1:self.sequence_length+1])

        return (x,y_true)
