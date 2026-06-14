import os
import joblib
import pandas as pd
import numpy as np

from backend.utils.feature_engineering import create_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load models once at startup
rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
gb = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
ada = joblib.load(os.path.join(MODEL_DIR, "adaboost.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

meta_model = joblib.load(
    os.path.join(MODEL_DIR, "stacked_meta_model.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)

threshold = joblib.load(
    os.path.join(MODEL_DIR, "best_threshold.pkl")
)


gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

smoking_map = {
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5
}


def predict_diabetes(data: dict):

    df = pd.DataFrame([data])

    df["gender"] = df["gender"].str.title()
    df["gender"] = df["gender"].map(gender_map)
    df["smoking_history"] = df["smoking_history"].map(smoking_map)

    # Create engineered features
    df = create_features(df)

    # Ensure all columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    numeric_cols = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ]

    df[numeric_cols] = scaler.transform(
        df[numeric_cols]
    )

    rf_prob = rf.predict_proba(df)[:, 1]
    gb_prob = gb.predict_proba(df)[:, 1]
    ada_prob = ada.predict_proba(df)[:, 1]
    xgb_prob = xgb.predict_proba(df)[:, 1]

    meta_features = np.column_stack([
        rf_prob,
        gb_prob,
        ada_prob,
        xgb_prob
    ])

    final_prob = meta_model.predict_proba(
        meta_features
    )[:, 1][0]

    prediction = int(final_prob >= threshold)

    return {
        "prediction": prediction,
        "diabetes": bool(prediction),
        "probability": round(float(final_prob), 4),
        "threshold": round(float(threshold), 4)
    }