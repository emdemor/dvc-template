import argparse
import yaml
from typing import Text

import pandas as pd
from base import *
from xtlearn.utils import make_directory, dump_pickle, load_pickle

from os.path import join

from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, r2_score


def testing(config_path: Text) -> None:
    """Load raw data

    Args:
        config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    log_target = config["feature_transform"]["log_target"]
    eval_mae = config["test"]["mean_absolute_error"]
    eval_r2 = config["test"]["r2_score"]

    model = load_pickle("pkl/model.pkl")
    X_test = load_pickle("pkl/features_test.pkl")
    y_test = load_pickle("pkl/targets_test.pkl")

    y_pred = model.predict(X_test)

    if log_target:
        y_pred = np.exp(y_pred)
        y_test = np.exp(y_test)

    metrics = []

    if eval_mae:
        metrics.append({"score": "mean_absolute_error", "value": mae(y_test, y_pred)})

    if eval_r2:
        metrics.append({"score": "R2", "value": r2_score(y_test, y_pred)})

    pd.DataFrame(metrics).to_csv("data/metrics.csv", index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    testing(config_path=args.config)
