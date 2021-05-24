import argparse
import yaml
from typing import Text

import pandas as pd
from xtlearn.utils import make_directory, dump_pickle, load_pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score


def splitting(config_path: Text) -> None:
    """Load raw data

    Args:
        config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    test_size = config["splitting"]["test_size"]
    random_state = config["splitting"]["random_state"]

    X = pd.read_csv("stages/X.csv")
    y = pd.read_csv("stages/y.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv("stages/X_train.csv", index=False)
    X_test.to_csv("stages/X_test.csv", index=False)
    y_train.to_csv("stages/y_train.csv", index=False)
    y_test.to_csv("stages/y_test.csv", index=False)

    # model = reg = GradientBoostingRegressor(
    #     loss="huber",
    #     random_state=42,
    #     n_estimators=290,
    #     alpha=0.9,
    # )

    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)

    # print("mae:", mae(y_test, y_pred))
    # print("R2:", r2_score(y_test, y_pred))
    # dump_pickle(model, "stages/model.pkl")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    splitting(config_path=args.config)
