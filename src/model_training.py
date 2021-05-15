import argparse
import yaml
from typing import Text

import pandas as pd
from base import *
from xtlearn.utils import make_directory, dump_pickle, load_pickle

from os.path import join

from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error as mae, r2_score


def splitting(config_path: Text) -> None:
    """Load raw data

    Args:
        config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    regressor = config["model"]["regressor"]

    X_train = load_pickle("pkl/features_train.pkl")
    y_train = load_pickle("pkl/targets_train.pkl")

    if regressor == "GradientBoostingRegressor":
        loss = config["model"]["loss"]
        random_state = config["model"]["random_state"]
        n_estimators = config["model"]["n_estimators"]
        alpha = config["model"]["alpha"]

        model = GradientBoostingRegressor(
            loss=loss,
            random_state=random_state,
            n_estimators=n_estimators,
            alpha=alpha,
        )

    model.fit(X_train, y_train)

    dump_pickle(model, "pkl/model.pkl")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    splitting(config_path=args.config)
