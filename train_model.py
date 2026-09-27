"""Train a simple, reproducible demonstration classifier from a CSV file.

Expected columns: age, bmi, previous_injury, family_history, joint_pain,
activity_limit, heavy_work, smoking, oa_risk.
All marker columns should be 0/1 and oa_risk should be the binary outcome.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "age", "bmi", "previous_injury", "family_history", "joint_pain",
    "activity_limit", "heavy_work", "smoking",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/oa_risk_model.joblib")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing = set(FEATURES + ["oa_risk"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data["oa_risk"].nunique() < 2:
        raise ValueError("oa_risk must contain both 0 and 1 classes")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(data[FEATURES], data["oa_risk"])
    # The app uses this attribute to build the prediction row safely.
    model.feature_names_in_ = FEATURES
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Saved model to {output} using {len(data)} rows.")


if __name__ == "__main__":
    main()
