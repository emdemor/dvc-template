import argparse
import yaml
import pandas as pd
from typing import Text
from os.path import join
from xtlearn.utils import make_directory, dump_pickle, load_pickle

from sklearn.pipeline import Pipeline
from xtlearn.feature_selection import FeatureSelector
from xtlearn.preprocessing import OneHotMissingEncoder, MeanModeImputer, MinMaxScaler

import numpy as np
from unidecode import unidecode
import pandas as pd


def format_value(x):
    x = str(x)

    if "," in x:
        return float(x.replace(".", "").replace(",", "."))
    else:
        try:
            return float(x)
        except:
            return np.nan


def format_bedrooms(x):
    text = str(x).strip()[0:2]
    if text != "na":
        return int(text)


def format_area_util(x):
    x = str(x).replace("M²", "")
    try:
        return float(x)
    except:
        return np.nan


def format_condominio(x):
    x = str(x)
    x = x.split("<s")[0].strip().replace(".", "").replace(",", ".")
    return float(x)


def format_column_names(x):
    return (x, unidecode(x).replace(" ", "_").lower())


def format_bairro(x):
    x = unidecode(x).lower().strip()
    return x


def format_area(x):
    try:
        x = float(x)
    except:
        x = np.nan
    return x


def get_columns_min_notna(data, min_notna=100):
    df_missings = pd.DataFrame(
        [{"column": col, "n_missings": data[col].notna().sum()} for col in data.columns]
    )
    cols = df_missings.loc[df_missings["n_missings"] >= min_notna, "column"]

    return data[cols]


def main(config_path):

    # Import parameters
    config = yaml.safe_load(open(config_path))

    raw_data_path = config["feature_transform"]["raw_data_path"]
    log_target = config["feature_transform"]["log_target"]
    columns = config["feature_transform"]["features"]
    columns_encode = config["feature_transform"]["features_to_encode"]

    df = pd.read_csv(raw_data_path)

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

    make_directory("stages")

    dump_pickle(pipe, join("stages", "feature_transform_pipeline.pkl"))
    X.to_csv(join("stages", "X.csv"), index=False)
    y.to_csv(join("stages", "y.csv"), index=False)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()
    main(config_path=args.config)
