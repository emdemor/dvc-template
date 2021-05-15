import argparse
import yaml
import pandas as pd
from sklearn.datasets import load_iris
from typing import Text
from xtlearn.utils import make_directory, dump_pickle, load_pickle

from base import *
from os.path import join

from sklearn.pipeline import Pipeline
from xtlearn.feature_selection import FeatureSelector
from xtlearn.preprocessing import OneHotMissingEncoder, MeanModeImputer, MinMaxScaler


def transform(config_path: Text) -> None:
    """Load raw data

    Args:
        config_path {Text}: path to config
    """

    config = yaml.safe_load(open(config_path))
    log_target = config["feature_transform"]["log_target"]
    columns = config["feature_transform"]["features"]
    columns_encode = config["feature_transform"]["features_to_encode"]

    df = load_pickle(join("pkl", "raw_data.pkl"))

    df["classe"] = (
        df["classe"]
        .str.replace("Casas", "Casa")
        .str.replace("Apartamentos", "Apartamento")
        .str.replace("Chácaras", "Rural")
        .str.replace("Casa em Condomínio para Venda", "Casa")
        .str.replace("Sobrado para Venda", "Casa")
        .str.replace("Sitio", "Rurais")
        .str.replace("Galpao", "Comercial")
        .str.replace("Ponto", "Comercial")
        .str.replace("Loja", "Comercial")
        .str.replace("Barracao", "Rurais")
        .str.replace("Sala", "Rurais")
        .str.replace("Terrenos", "Terreno")
        .str.replace("Rural", "Rurais")
        .str.replace("Chacara", "Rurais")
        .str.replace("Area", "Rurais")
        .str.replace("Fazenda", "Rurais")
    )
    df["bairro"] = df["bairro"].apply(format_bairro)
    df["valor"] = df["valor"].apply(format_value)
    df["area_util"] = df["area_util"].apply(format_area)

    df_filtered = get_columns_min_notna(
        df, min_notna=100
    )  # .drop(columns = drop_columns)

    df_filtered = df_filtered[
        ~(df_filtered["classe"].isin(["Terrenos"])) & (df_filtered["valor"].notna())
    ]

    X = df_filtered.drop(columns="valor")

    y = df_filtered.get("valor")

    if log_target:
        y = np.log(y)

    pipe = Pipeline(
        steps=[
            ("selector", FeatureSelector(features=columns)),
            ("imputer", MeanModeImputer()),
            (
                "encoder",
                OneHotMissingEncoder(columns=columns_encode),
            ),
            ("scaler", MinMaxScaler()),
        ]
    ).fit(X, y)

    X = pipe.transform(X)

    dump_pickle(X, join("pkl", "features.pkl"))
    dump_pickle(y, join("pkl", "targets.pkl"))


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    transform(config_path=args.config)
