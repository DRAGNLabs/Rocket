# Rocket

Rocket is a modified version of the Llama 2 architecture, implemented through PyTorch Lightning to easily enable distributed training on a number of configurations.

## Project Structure

This repository consists of:

- **configs**: the configuration folder holding all configs for use in training, data preparation, and evaluation.
- **dataset**: the dataset folder should store all raw and tokenized data, as well as tokenizers.
- **runs**: contains all results from training and evaluation jobs.
- **slurm**: slurm scripts for various tasks/=.
- **tokenizer**: various scripts pertaining to tokenization, as well as the core tokenizer class in [tokenizer.py](./tokenizer/tokenizer.py).
- **utils**: various utils.
- **dataset.py**: containing PyTorch Lightning DataModule class and DataSet class. These classes should be modified for specific use cases.
- **inference.py**: Script for running inference, given a configuration.
- **llama.py**: Core LightningModule class for Llama.
- **model.py**: Model code for Llama.
- **setup.py**: Setup script specifically constructed for obtaining and formatting OpenOrca data for training.
- **tokenize_data.py**: Tokenizes data found in corresponding path in given config.
- **train.py**: Training script.

## Setting up Rocket Llama

### Environment

Create a Mamba environment with python=3.9, preferably named ```rocket```:
```mamba create -n rocket python=3.9```

If it is named differently, the environment activation commands in the Slurm scripts must be changed.

Run ```pip install -r requirements.txt```.

### Setting up a Config

Configuration YAML (Yet Another Markdown Language) files are used to define all paths, settings, and hyperparameters for training tokenizers, tokenizing data, training models, and running inference on models. In the config folder, you can create a new config by copying default_config.yaml. Fill out the class parameters accordingly. Or, edit the parameters in default_config.yaml directly.

In your config yaml, enter the necessary information in the absolute paths for datasets, the tokenizer, and the root directory. The default names for the tokenizer and dataset will work with the setup script.

### Getting OpenOrca Data

This repo was developed to use data from [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca). If using other data, it is up to the user to prepare the data correctly. Ultimately, the paths for training and evaluation must point to pickle files containing the data.

To obtain the OpenOrca data, run [setup.py](./setup.py). This will download two parquet files into your dataset folder. The script will then consolidate both parquet files into a single parquet file(for training), as well as a csv file(for training the tokenizer).

### Preparing Tokenizer

Llama is designed to use [SentencePiece](https://github.com/google/sentencepiece). To prepare the tokenizer, you can either:

- Train a new tokenizer from scratch based on your data.
- Use the original Llama 2 tokenizer trained by Meta.

#### Training SentencePiece Tokenizer from scratch

A SentencePiece tokenizer can be trained by running [train_tokenizer.sh](./slurm/train_tokenizer.sh). This script is simply a wrapper for the SentencePiece python module; it seems easier than building and installing SentencePiece from source. Pass in all arguments in quotations, ex:

```python3 train_tokenizer.py "--input=../dataset/raw/openorca_combined.csv --input_format=text --input_sentence_size=1000000 --train_extremely_large_corpus=true --model_prefix=tokenizer --vocab_size=32000 --shuffle_input_sentence=true --pad_id=3""```

You can adjust the vocabularly size with `--vocab_size`.

You will want to verify that the [Tokenizer](./tokenizer/tokenizer.py) class is using ```.pad_id()``` as opposed to a custom pad string, i.e. "['<pad>']".

Then, submit the job:
```sbatch train_tokenizer.sh```

You can find further information on training arguments in the SentencePiece documentation: 
- [SentencePiece Repository](https://github.com/google/sentencepiece)
- [Training options](https://github.com/google/sentencepiece/blob/master/doc/options.md)

#### Using Original Llama 2 Tokenizer

To obtain the original Llama 2 Tokenizer, [Request access for Llama 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

Clone the [repository](https://github.com/facebookresearch/llama).

When download link has been obtained via email, run `./download.sh` in the llama repo.

When asked, paste the url sent to your email.

Once downloaded, move tokenizer.model into Tokenizers folder of Rocket repo.

Move dataset file(s) into `/Dataset/raw`

The tokenizer being used utilizes sentencepiece. By default, sentencepiece uses -1 as the id for padding tokens, meaning padding is disabled by default. This causes problems if you want to use a padding token. To add a new token representing padding, you can run [add_tokens.py](./tokenizer/add_tokens.py) after putting the string `<pad>` into the special_tokens list; this should already be present. Additionally, you will need to specify the path to the tokenzier within this script. The new tokenizer will have the additional padding token. Then, in [tokenizer.py](./tokenizer/tokenizer.py), ensure that `pad_id` in the tokenizer class is set to the string you defined for padding, rather than the SentencePieceProcessor `pad_id`.

### Tokenizing data

To tokenize data, ensure that the correct tokenizer you wish to use is specified in the config file. Navigate to [tokenize_data.sh](./slurm/tokenize_data.sh) and verify that your desired config file is being passed as a parameter in the script. 

Then, submit the job:
```sbatch tokenize_data.sh```

[tokenize_data.py](./tokenize_data.py) will tokenize the given data files as defined in the config yaml file, according to the tokenizer path given. This script expects raw data to be in parquet file format by default, but this could be changed.


## Training

The [train.py](./train.py) takes as an argument a path to a config yaml file. There is a slurm script, [run_train.sh](./slurm/run_train.sh) that calls this script. Edit the slurm script to use your config file, and training will begin when ran.
