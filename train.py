import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess

# LOAD DATA
df = pd.read_csv("European_Bank.csv")

# PREPROCESS
df = preprocess(df)

# SPLIT
X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# EVALUATION
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# SAVE MODEL
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")