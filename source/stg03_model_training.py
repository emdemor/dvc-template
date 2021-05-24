import argparse
import yaml
from typing import Text

import pandas as pd
from xtlearn.utils import make_directory, dump_pickle, load_pickle

from os.path import join

from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error as mae, r2_score


def training(config_path: Text) -> None:
    """Load raw data

    Args:
        config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    regressor = config["model"]["regressor"]

    X_train = pd.read_csv("stages/X_train.csv")
    y_train = pd.read_csv("stages/y_train.csv").iloc[:, 0]

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

    dump_pickle(model, "stages/model.pkl")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    training(config_path=args.config)

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    training(config_path=args.config)
