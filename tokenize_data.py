# Change this to tokenization script you want to use
from tokenizer import wikitext_tokenization as tokenizer
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import sys
import yaml
from utils.data_utils import Struct
from sklearn.model_selection import train_test_split

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

    # Tokenize one whole file, and then do train test split
    if config.raw_dataset_path:
        raw_data = config.raw_dataset_path

        # Load Dataset into pd.DataFrame
        training_dataframe:pd.DataFrame = pd.read_csv(raw_data, dtype=str, na_filter=False)#.iloc[:25]
        #training_dataframe:pd.DataFrame = pd.read_parquet(raw_data)

        # Generate tokenized file
        tokenized_df:pd.DataFrame = tokenizer.generate_tokenized_file(training_dataframe, tokenizer_path=config.tokenizer_path)

        # Split into train/val/test 85/10/5
        train, test = train_test_split(tokenized_df, test_size=0.15, random_state=config.seed)
        val, test = train_test_split(test, test_size=0.25, random_state=config.seed)

        # Save train, validation, and test to pickle files
        out_dir_train = Path(config.train_path)
        out_dir_val = Path(config.val_path)
        out_dir_test = Path(config.test_path)

        if not out_dir_train.parent.exists():
            out_dir_train.parent.mkdir(parents=True)

        if not out_dir_val.parent.exists():
            out_dir_val.parent.mkdir(parents=True)

        if not out_dir_test.parent.exists():
            out_dir_test.parent.mkdir(parents=True)

        train.to_pickle(out_dir_train.parent / out_dir_train.name)
        val.to_pickle(out_dir_val.parent / out_dir_val.name)
        test.to_pickle(out_dir_test.parent / out_dir_test.name)

        print(f'\033[0;37m Saved train, validation, and test as pickle files at "{out_dir_train.parent}"')    
        print(f"# of tokenized prompts: {len(tokenized_df)}\n")

    # Tokenize train, test, and validation set seperately
    elif config.raw_train_path and config.raw_test_path and config.raw_val_path:
        raw_train = config.raw_train_path
        raw_test = config.raw_test_path
        raw_val = config.raw_val_path

        # Load Dataset into pd.DataFrame
        training_dataframe:pd.DataFrame = pd.read_csv(raw_train, dtype=str, na_filter=False)
        test_dataframe:pd.DataFrame = pd.read_csv(raw_test, dtype=str, na_filter=False)
        val_dataframe:pd.DataFrame = pd.read_csv(raw_val, dtype=str, na_filter=False)

        # Generate tokenized file
        tokenized_train:pd.DataFrame = tokenizer.generate_tokenized_file(training_dataframe, tokenizer_path=config.tokenizer_path)
        tokenized_test:pd.DataFrame = tokenizer.generate_tokenized_file(test_dataframe, tokenizer_path=config.tokenizer_path)
        tokenized_val:pd.DataFrame = tokenizer.generate_tokenized_file(val_dataframe, tokenizer_path=config.tokenizer_path)

        # Save train, validation, and test to pickle files
        out_dir_train = Path(config.train_path)
        out_dir_val = Path(config.val_path)
        out_dir_test = Path(config.test_path)

        if not out_dir_train.parent.exists():
            out_dir_train.parent.mkdir(parents=True)

        if not out_dir_val.parent.exists():
            out_dir_val.parent.mkdir(parents=True)

        if not out_dir_test.parent.exists():
            out_dir_test.parent.mkdir(parents=True)

        tokenized_train.to_pickle(out_dir_train.parent / out_dir_train.name)
        tokenized_val.to_pickle(out_dir_val.parent / out_dir_val.name)
        tokenized_test.to_pickle(out_dir_test.parent / out_dir_test.name)

        print(f'\033[0;37m Saved train, validation, and test as pickle files at "{out_dir_train.parent}"')    
        print(f"# of tokenized prompts in train: {len(tokenized_train)}\n")
        print(f"# of tokenized prompts in validation: {len(tokenized_val)}\n")
        print(f"# of tokenized prompts in test: {len(tokenized_test)}\n")

if __name__== "__main__":
    main()