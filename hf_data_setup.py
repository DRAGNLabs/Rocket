import datasets
from datasets import load_dataset
import sys
import yaml
from utils.data_utils import Struct
from pathlib import Path

def main(config):
    dataset_directory = Path(config.dataset_directory)
    # Check if directory exists
    if not dataset_directory.exists():
        dataset_directory.mkdir(parents=True)
    dataset = load_dataset(config.hf_dataset_name, name=config.hf_dataset_config)

    # check if dataset is of type datasets.arrow_dataset.Dataset
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        filename = args.dataset_subset + '.csv'
        dataset.to_csv(dataset_directory / filename)
    elif isinstance(dataset, datasets.dataset_dict.DatasetDict):
        for key, value in dataset.items():
            filename = key + '.csv'
            value.to_csv(dataset_directory / filename)
    else:
        print('Dataset is not of type datasets.arrow_dataset.Dataset or datasets.dataset_dict.DatasetDict')


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print('Usage: python hf_data_setup.py <config_path>')
        exit()
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)
    main(config)