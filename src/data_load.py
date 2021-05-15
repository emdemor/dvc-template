import argparse
import yaml
import pandas as pd
from sklearn.datasets import load_iris
from typing import Text
from xtlearn.utils import make_directory, dump_pickle
from os.path import join


def data_load(config_path):
    """Load raw data

    Args:
        config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    raw_data_path = config["data_load"]["raw_data_path"]

    data = pd.read_csv(raw_data_path)

    make_directory("pkl")

    dump_pickle(data, join("pkl", "raw_data.pkl"))


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    data_load(config_path=args.config)
