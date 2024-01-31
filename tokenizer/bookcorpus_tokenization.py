import pandas as pd
from .tokenizer import Tokenizer

def tokenize_data_chunk_wikitext(tokenizer, chunk):  
    '''
    Take some tokenizer object and some dictionary-like(?) data format
    ''' 
    to_tokenize:str = chunk['text']
    chunk['Tokenized_Data'] = tokenizer.encode(to_tokenize, bos=True, eos=True)

    # print(chunk.columns)
    return chunk

def generate_tokenized_file(df:pd.DataFrame, tokenizer_path):
    # Call 'tokenize_data_chunk' over entire file
    tokenizer = Tokenizer(tokenizer_path)
    tok_lambda = lambda x: tokenize_data_chunk_wikitext(tokenizer=tokenizer, chunk=x)  # 'df.' of line 62 becomes 'x' in this lambda
    print(f'Dataframe: {df}\n\n')
    df1 = df.progress_apply(tok_lambda, axis=1)
    df1 = df1.drop(['text'], axis=1)
    return df1