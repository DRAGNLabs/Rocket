# Generation with a single string
import sys
import yaml
import os
import torch
from typing import List

from model import Model
from tokenizer.tokenizer import Tokenizer
from utils.data_utils import Struct

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

def generate(
    model,
    tokenizer,
    prompt: str,
    max_gen_len: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
) -> List[str]:
    
    prompt_tokens = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False)).reshape(1,-1)
    generate_ids = model.generate(prompt_tokens.to(device), 
                                  max_length=max_gen_len, 
                                  temperature=temperature, 
                                  top_p=top_p, 
                                  repetition_penalty=repetition_penalty, 
                                  do_sample=True)
    generate_tokens = generate_ids.tolist()
    decoded = tokenizer.decode(generate_tokens)
    return decoded

def generation(config):
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

    model.cuda()
    model.eval()
    
    with open(config.generation_path, 'r') as f:
        prompt = f.read()

    decoded = generate(model,
                        tokenizer,
                        prompt,
                        max_gen_len = config.max_gen_len,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        repetition_penalty=config.repetition_penalty,)

    print('decoded: ', decoded)

    print('\nNo errors!\n')

def main():
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    generation(config)

if __name__ == "__main__":
    main()