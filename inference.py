import sys
import yaml
import os
import torch
from tqdm import tqdm
from typing import List

from dataset import DataSet
from torch.utils.data import DataLoader
from model import Model
from tokenizer.tokenizer import Tokenizer
from utils.data_utils import Struct

from nltk.translate.chrf_score import corpus_chrf
from nltk.translate.bleu_score import corpus_bleu

#TODO: this script kinda works but not really. Need to:
# - Determine metrics to use
# - Left or right padding

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

def to_device(tensors):
    return [i.to(device) for i in tensors]

def save_output(path, data):
    with open(path, 'w') as f:
        for line in data:
            f.write(line + '\n')

def generate(
    model,
    tokenizer,
    dataset: DataSet,
    batch_size: int,
    max_gen_len: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    save_path: str = None,
) -> List[str]:
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            collate_fn=dataset.pad_to_longest)
    
    model = model.to(device)
    model.eval()

    tgts = []
    preds = []
    with torch.no_grad():
        val_bar = tqdm(dataloader, desc='Running inference set')
        for data in val_bar:
            src, src_mask, tgt = data

            src, src_mask, tgt = to_device([src, src_mask, tgt])

            output_ids = model.generate(input_ids=src,
                                        attention_mask=src_mask,
                                        num_beams=5,
                                        min_length=0,
                                        max_new_tokens=max_gen_len)
            
            tgts += tokenizer.decode(tgt.tolist())
            preds += tokenizer.decode(output_ids.tolist())

        # Get chrf score
        chrf = corpus_chrf(tgts, preds)

        # Get bleu score
        bleu = corpus_bleu([[tgt] for tgt in tgts], preds)
        scores = ['chrf: ' + str(chrf), 'bleu: ' + str(bleu)]

    if save_path:
        if not os.path.exists(save_path): os.makedirs(save_path)
        save_output(save_path + '/tgts.txt', tgts)
        save_output(save_path + '/preds.txt', preds)
        save_output(save_path + '/scores.txt', scores)

    return tgts, preds, (chrf, bleu)

def inference(config):
    print('Beginning Inference')
    
    tokenizer = Tokenizer(model_path=config.tokenizer_path)  # including this for the special tokens (i.e. pad)
    config.vocab_size = tokenizer.n_words
    config.pad_id = tokenizer.pad_id

    # Build model class
    model = Model(tokenizer=tokenizer, config=config)

    # Load checkpoint
    checkpoint_path=config.checkpoint_path

    print(f"Using checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])

    model = model.model
    
    inference_dataset_path = config.inference_dataset_path
    inference_dataset = DataSet(inference_dataset_path,
                                       tokenizer.pad_id, 
                                       tokenizer.bos_id, 
                                       tokenizer.eos_id, 
                                       config.max_sequence_embeddings)

    tgts, preds, (chrf, bleu) = generate(model=model,
                                   tokenizer=tokenizer,
                                   dataset=inference_dataset,
                                   batch_size=config.batch_size,
                                   max_gen_len = config.max_gen_len,
                                   repetition_penalty=config.repetition_penalty,
                                   save_path=config.inference_output_path)

    print('tgts: ', tgts)
    print('preds: ', preds)
    print('chrf: ', chrf)
    print('bleu: ', bleu)

    print('\nNo errors!\n')

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    inference(config)

if __name__ == "__main__":
    main()