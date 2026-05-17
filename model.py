import pickle

def load_model():
    return pickle.load(open("model.pkl", "rb"))

def predict(model, df):
    df = df.copy()

    if "Exited" in df.columns:
        X = df.drop("Exited", axis=1)
    else:
        X = df.copy()

    # HARD FIX: feature alignment (Geography issue solved here)
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    df["ChurnProb"] = model.predict_proba(X)[:, 1]
    return df