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
    print(dataset)
    for key, value in dataset.items():
        filename = key + '.parquet'
        value.to_parquet(dataset_directory / filename)

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print('Usage: python generate_raw_dataset.py <config_path>')
        exit()
    config_path = args[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert args dict to object
    config = Struct(**config)
    main(config)