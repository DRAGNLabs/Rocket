import os
from logging import getLogger
from typing import List
from sentencepiece import SentencePieceProcessor
import pandas as pd
from tokenizer import Tokenizer

def tokenize_data_chunk_orca(tokenizer, chunk):  
    '''
    Take some tokenizer object and some dictionary-like(?) data format
    ''' 
    to_tokenize:str = chunk['system_prompt'] + '<SEP>' + chunk['question'] + '<SEP>' + chunk['response']
    chunk['Tokenized_Data'] = tokenizer.encode(to_tokenize, bos=True, eos=True)

    # print(chunk.columns)
    return chunk

def generate_tokenized_file(df:pd.DataFrame, tokenizer_path):
    # Call 'tokenize_data_chunk' over entire file
    tokenizer = Tokenizer(tokenizer_path)
    tok_lambda = lambda x: tokenize_data_chunk_orca(tokenizer=tokenizer, chunk=x)  # 'df.' of line 62 becomes 'x' in this lambda
    print(f'Dataframe: {df}\n\n')
    df1 = df.progress_apply(tok_lambda, axis=1)
    df1 = df1.drop(['system_prompt','question','response'], axis=1) # Drop everything but id and tokenized_data
    return df1