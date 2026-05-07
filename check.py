import pandas as pd
import joblib

clf_features = joblib.load("classification_features.pkl")
X_predict = pd.read_excel("X_predict_preprocessed.xlsx")

# Check if all features match
missing = [f for f in clf_features if f not in X_predict.columns]
extra   = [f for f in X_predict.columns if f not in clf_features]

print("Missing from X_predict:", missing)
print("Extra in X_predict:", extra)
print("Shape:", X_predict.shape)
