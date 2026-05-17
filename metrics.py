import numpy as np

def compute_rsi(df):
    return (
        df["IsActiveMember"] * 0.4 +
        (df["NumOfProducts"] / df["NumOfProducts"].max()) * 0.3 +
        df["HasCrCard"] * 0.1 +
        (df["Balance"] / df["Balance"].max()) * 0.2
    )