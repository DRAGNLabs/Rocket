from tokenizer import tokenizer
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import os
import sys
import yaml
from utils.data_utils import Struct
from sklearn.model_selection import train_test_split
import os

# To use: define raw_dataset_path and tokenized_dataset_path in config
def main():
    tqdm.pandas()
    args = sys.argv
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)

    print('\nStarting tokenization...\n')
    
    raw_data = config.raw_dataset_path

    # Load Dataset into pd.DataFrame
    #training_dataframe:pd.DataFrame = pd.read_csv(raw_data, dtype=str, na_filter=False)#.iloc[:25]
    training_dataframe:pd.DataFrame = pd.read_parquet(raw_data)

    # Generate tokenized file
    tokenized_df:pd.DataFrame = tokenizer.generate_tokenized_file(training_dataframe, tokenizer_path=config.tokenizer_path, seq_len=config.sequence_length)

    # Split into train/val/test 85/10/5
    train, test = train_test_split(tokenized_df, test_size=0.15, random_state=config.seed)
    val, test = train_test_split(test, test_size=0.25, random_state=config.seed)

    # Save train, validation, and test to pickle files
    out_dir_train = Path(config.train_path)
    out_dir_eval = Path(config.eval_path)
    out_dir_test = Path(config.test_path)

    if not out_dir_train.parent.exists():
        out_dir_train.parent.mkdir(parents=True)

    if not out_dir_eval.parent.exists():
        out_dir_eval.parent.mkdir(parents=True)

    if not out_dir_test.parent.exists():
        out_dir_test.parent.mkdir(parents=True)

    train.to_pickle(out_dir_train.parent / out_dir_train.name)
    val.to_pickle(out_dir_eval.parent / out_dir_eval.name)
    test.to_pickle(out_dir_test.parent / out_dir_test.name)

    print(f'\033[0;37m Saved train, validation, and test as pickle files at "{out_dir_train.parent}"')    
    print(f"# of tokenized prompts: {len(tokenized_df)}\n")

if __name__== "__main__":
    main()