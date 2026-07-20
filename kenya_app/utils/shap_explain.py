"""
SHAP explanations for the Stage 1 model.

Earlier notebook work used TreeExplainer on a substitute Random Forest,
because the deployed model at the time (Logistic Regression) isn't
tree-structured. Here we explain the actual deployed model directly via
shap.LinearExplainer - higher fidelity, and it stops being a substitute
explanation once the deployed model is genuinely linear. If a future model
swap makes the deployed Stage 1 model non-linear again, build_explainer()
below falls back to a general-purpose shap.Explainer automatically.
"""

import numpy as np
import shap
import streamlit as st

from utils.model_loader import get_estimator


def _feature_names(preprocessor, categorical_features, numeric_features):
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    return list(cat_names) + list(numeric_features)


def _transform(preprocessor, df, categorical_features, numeric_features):
    X = preprocessor.transform(df[categorical_features + numeric_features])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


@st.cache_resource(show_spinner=False)
def build_explainer(_stage1_pipeline, _background_df, categorical_features, numeric_features):
    preprocessor = _stage1_pipeline.named_steps["preprocessor"]
    model = get_estimator(_stage1_pipeline)

    background = _background_df.sample(min(200, len(_background_df)), random_state=42)
    background_t = _transform(preprocessor, background, categorical_features, numeric_features)

    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, background_t)
    elif hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, background_t)

    feature_names = _feature_names(preprocessor, categorical_features, numeric_features)
    return explainer, feature_names


def explain_row(stage1_pipeline, explainer, feature_names, row_df, categorical_features, numeric_features, top_n=8):
    preprocessor = stage1_pipeline.named_steps["preprocessor"]
    X_t = _transform(preprocessor, row_df, categorical_features, numeric_features)

    if hasattr(explainer, "shap_values"):
        raw = explainer.shap_values(X_t)
        values = raw[1] if isinstance(raw, list) else np.array(raw)
        if values.ndim == 3:
            values = values[:, :, 1]
    else:
        values = explainer(X_t).values

    row_values = values[0]
    import pandas as pd
    result = pd.DataFrame({"Feature": feature_names, "SHAP Value": row_values})
    result["Abs"] = result["SHAP Value"].abs()
    result = result.sort_values("Abs", ascending=False).head(top_n).drop(columns="Abs")
    return result
