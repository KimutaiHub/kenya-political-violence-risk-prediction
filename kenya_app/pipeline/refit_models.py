"""
Re-export the deployed models under a numpy<2 / scikit-learn<1.4 environment.

Why this exists: numpy 2.0 changed internal module paths (numpy._core did
not exist before 2.0). A model pickled under numpy>=2.0 cannot be loaded by
numpy<2.0 - which matters here because Python 3.8 cannot install numpy>=2.0
at all, so any teammate on Python 3.8 needs models pickled under numpy<2.0.

This script re-FITS (not just re-pickles) the models, because a pickle
created under one numpy/sklearn version generally can't be safely loaded
under a different one to begin with - the only really safe fix is to fit
fresh under the target environment. It uses the exact hyperparameters
already selected during the notebook's hyperparameter search, so the
resulting models are functionally identical (same algorithm, same tuned
params, same random_state) - only the serialization environment changes.

Usage (see README, "Model version compatibility" for the venv setup):
    python pipeline/refit_models.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
MODELS_DIR = APP_ROOT / "models"

RANDOM_STATE = 42

# These hyperparameters come directly from the notebook's Section 6.9
# hyperparameter search (RandomizedSearchCV best_params_). Update this if
# the notebook's winning model or hyperparameters ever change.
STAGE1_PARAMS = dict(C=10.0, class_weight="balanced", max_iter=1000, penalty="l2", solver="lbfgs")
STAGE2_PARAMS = dict(
    n_estimators=200, max_depth=16, min_samples_leaf=5, min_samples_split=2,
    max_features="sqrt", class_weight="balanced", n_jobs=-1,
)


def make_preprocessor(categorical_features, numeric_features):
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(transformers=[
        ("cat", encoder, categorical_features),
        ("num", StandardScaler(), numeric_features),
    ], remainder="drop")


def main():
    print(f"Refitting under numpy {np.__version__}, expecting <2.0")
    if int(np.__version__.split(".")[0]) >= 2:
        print(
            "WARNING: numpy >= 2.0 is installed. This defeats the purpose of this script - "
            "run it inside a numpy<2 virtualenv instead (see README).", file=sys.stderr,
        )

    df = pd.read_parquet(DATA_DIR / "kenya_conflict_full.parquet")
    df["WEEK"] = pd.to_datetime(df["WEEK"])

    feature_lists = joblib.load(MODELS_DIR / "model_feature_lists.joblib")
    categorical_features = feature_lists["categorical_features"]
    numeric_features = feature_lists["numeric_features"]

    df["IS_FATAL"] = (df["FATALITIES"] > 0).astype(int)

    def severity_3class(x):
        if x <= 2:
            return "Low"
        elif x <= 5:
            return "Medium"
        return "High"

    df["FATALITY_SEVERITY"] = df["FATALITIES"].apply(severity_3class)

    X = df[categorical_features + numeric_features].copy()
    train_mask = df["YEAR"] < 2022

    X_train = X.loc[train_mask].copy()
    y_stage1_train = df.loc[train_mask, "IS_FATAL"].copy()

    fatal_train_mask = train_mask & (df["FATALITIES"] > 0)
    X_stage2_train = X.loc[fatal_train_mask].copy()
    y_stage2_train = df.loc[fatal_train_mask, "FATALITY_SEVERITY"].copy()

    stage2_label_encoder = LabelEncoder()
    stage2_label_encoder.fit(y_stage2_train)

    stage1_pipeline = Pipeline(steps=[
        ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
        ("model", LogisticRegression(random_state=RANDOM_STATE, **STAGE1_PARAMS)),
    ])
    stage1_pipeline.fit(X_train, y_stage1_train)
    print("Stage 1 refit complete.")

    stage2_pipeline = Pipeline(steps=[
        ("preprocessor", make_preprocessor(categorical_features, numeric_features)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, **STAGE2_PARAMS)),
    ])
    stage2_pipeline.fit(X_stage2_train, y_stage2_train)
    print("Stage 2 refit complete.")

    joblib.dump(stage1_pipeline, MODELS_DIR / "stage1_final_model.joblib")
    joblib.dump(stage2_pipeline, MODELS_DIR / "stage2_final_model.joblib")
    joblib.dump(stage2_label_encoder, MODELS_DIR / "stage2_label_encoder.joblib")
    joblib.dump(feature_lists, MODELS_DIR / "model_feature_lists.joblib")

    print(f"\nSaved to {MODELS_DIR} under numpy {np.__version__}.")


if __name__ == "__main__":
    main()
