import pandas as pd
import numpy as np


def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)


def preprocess(df):

    # --------------------------------
    # DROP UNUSED COLUMNS
    # --------------------------------
    drop_cols = ["CustomerId", "Surname", "Year"]

    existing_cols = [col for col in drop_cols if col in df.columns]

    df = df.drop(existing_cols, axis=1)

    # --------------------------------
    # ENCODE CATEGORICAL VARIABLES
    # --------------------------------
    df = pd.get_dummies(
        df,
        columns=["Geography", "Gender"],
        drop_first=True
    )

    # --------------------------------
    # FEATURE ENGINEERING
    # --------------------------------

    # Engagement Score
    df["EngagementScore"] = (
        0.5 * df["IsActiveMember"] +
        0.3 * normalize(df["Tenure"]) +
        0.2 * normalize(df["NumOfProducts"])
    )

    # Product Utilization
    df["ProductUtilization"] = (
        0.7 * normalize(df["NumOfProducts"]) +
        0.3 * df["HasCrCard"]
    )

    # Financial Strength
    df["FinancialStrength"] = (
        0.6 * normalize(df["Balance"]) +
        0.4 * normalize(df["EstimatedSalary"])
    )

    # Relationship Strength Index
    df["RSI"] = (
        0.4 * df["EngagementScore"] +
        0.3 * df["ProductUtilization"] +
        0.3 * df["FinancialStrength"]
    )

    # Premium Customer
    threshold = df["Balance"].quantile(0.80)

    df["PremiumCustomer"] = np.where(
        df["Balance"] >= threshold,
        1,
        0
    )

    return df