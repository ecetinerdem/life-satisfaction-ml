import pandas as pd
from lifesat_ml.config import CONFIG

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Replace special values
    df = df.replace([-90, -88, -99], pd.NA)

    # Drop missing
    df = df.dropna()

    return df

def prepare_features(df):
    X = df.drop(columns=[CONFIG["target_column"]])
    y = df[CONFIG["target_column"]]
    return X, y