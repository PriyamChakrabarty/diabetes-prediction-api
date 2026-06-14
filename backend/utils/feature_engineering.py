import pandas as pd


def create_features(df):

    df["high_glucose"] = (
        df["blood_glucose_level"] >= 140
    ).astype(int)

    df["very_high_glucose"] = (
        df["blood_glucose_level"] >= 200
    ).astype(int)

    df["high_hba1c"] = (
        df["HbA1c_level"] >= 6.5
    ).astype(int)

    df["prediabetes_hba1c"] = (
        df["HbA1c_level"] >= 5.7
    ).astype(int)

    df["obese"] = (
        df["bmi"] >= 30
    ).astype(int)

    df["overweight"] = (
        df["bmi"] >= 25
    ).astype(int)

    df["senior_risk"] = (
        df["age"] >= 45
    ).astype(int)

    df["hypertension_risk"] = (
        df["hypertension"] == 1
    ).astype(int)

    df["heart_risk"] = (
        df["heart_disease"] == 1
    ).astype(int)

    df["smoker_risk"] = (
        df["smoking_history"].isin([1, 2, 3])
    ).astype(int)

    df["metabolic_risk"] = (
        (df["HbA1c_level"] >= 6.5)
        & (df["blood_glucose_level"] >= 140)
    ).astype(int)

    df["obese_age_risk"] = (
        (df["bmi"] >= 30)
        & (df["age"] >= 45)
    ).astype(int)

    df["cardio_diabetes_risk"] = (
        (df["hypertension"] == 1)
        | (df["heart_disease"] == 1)
    ).astype(int)

    df["severe_risk"] = (
        (df["blood_glucose_level"] >= 200)
        & (df["HbA1c_level"] >= 6.5)
        & (df["bmi"] >= 30)
    ).astype(int)

    return df