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
