import pandas as pd
from preprocess import preprocess

def load_raw():
    return pd.read_csv("European_Bank.csv")

def load_processed():
    df = load_raw()
    df = preprocess(df)
    return df