"""
Model loading, made robust and model-agnostic.

Two separate problems, two separate fixes:

1. "numpy._core" / InconsistentVersionWarning-class failures: these happen
   when a model is pickled under one numpy/scikit-learn major version and
   loaded under another. There is no way to make an old numpy load a
   numpy>=2.0 pickle - the fix has to happen at EXPORT time (pin numpy<2 and
   a matching scikit-learn when saving, see pipeline/refit_models.py). What
   this module does is make the FAILURE MODE graceful: if a load fails, the
   app should show a clear diagnostic instead of an unhandled crash, and the
   System Status page should be able to report it.

2. "Model swapping": nothing here should hardcode assumptions about which
   algorithm is deployed (Logistic Regression today, could be anything with
   a compatible sklearn interface tomorrow). Every access pattern below only
   assumes the loaded object exposes .predict / .predict_proba / .named_steps
   the way any sklearn Pipeline does - swap the .joblib file, nothing else
   needs to change.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import sklearn
import streamlit as st


@dataclass
class ModelLoadResult:
    model: object
    ok: bool
    error: Optional[str] = None
    path: Optional[str] = None


def _safe_load(path: Path) -> ModelLoadResult:
    if not path.exists():
        return ModelLoadResult(model=None, ok=False, error=f"File not found: {path}", path=str(path))
    try:
        model = joblib.load(path)
        return ModelLoadResult(model=model, ok=True, path=str(path))
    except ModuleNotFoundError as e:
        return ModelLoadResult(
            model=None, ok=False, path=str(path),
            error=(
                f"Version-incompatible model file ({e}). This model was almost certainly saved "
                f"under a different numpy/scikit-learn major version than what's installed here "
                f"(installed: numpy compatible with scikit-learn {sklearn.__version__}). "
                f"Re-export the model under a matching environment - see README, "
                f"'Model version compatibility'."
            ),
        )
    except Exception as e:
        return ModelLoadResult(model=None, ok=False, path=str(path), error=f"{type(e).__name__}: {e}")


@st.cache_resource(show_spinner=False)
def load_all_models(models_dir: str):
    models_dir = Path(models_dir)
    stage1 = _safe_load(models_dir / "stage1_final_model.joblib")
    stage2 = _safe_load(models_dir / "stage2_final_model.joblib")
    label_encoder = _safe_load(models_dir / "stage2_label_encoder.joblib")
    feature_lists = _safe_load(models_dir / "model_feature_lists.joblib")

    # A pickle can LOAD successfully across incompatible scikit-learn versions
    # and still fail at PREDICT time (e.g. a newer sklearn's DecisionTreeClassifier
    # expects an attribute like monotonic_cst that an older-sklearn-fitted tree
    # doesn't have). Catch that here too, at diagnostic time, rather than letting
    # a user hit it mid-prediction on the Event Scorer page.
    if stage1.ok and feature_lists.ok:
        stage1 = _predict_smoke_test(stage1, feature_lists.model)
    if stage2.ok and feature_lists.ok:
        stage2 = _predict_smoke_test(stage2, feature_lists.model)

    return {
        "stage1": stage1,
        "stage2": stage2,
        "label_encoder": label_encoder,
        "feature_lists": feature_lists,
    }


def _predict_smoke_test(result: ModelLoadResult, feature_lists: dict) -> ModelLoadResult:
    import numpy as np
    import pandas as pd

    try:
        cat_cols = feature_lists["categorical_features"]
        num_cols = feature_lists["numeric_features"]
        dummy = {c: ["placeholder"] for c in cat_cols}
        dummy.update({c: [0.0] for c in num_cols})
        dummy_row = pd.DataFrame(dummy)
        result.model.predict(dummy_row)
        return result
    except Exception as e:
        return ModelLoadResult(
            model=None, ok=False, path=result.path,
            error=(
                f"Model loaded but failed a live prediction smoke test ({type(e).__name__}: {e}). "
                f"This usually means the installed scikit-learn version doesn't match the one used "
                f"to fit the model, even though the pickle itself loaded without error. Match the "
                f"scikit-learn pin in requirements.txt, or re-run pipeline/refit_models.py under "
                f"this environment."
            ),
        )


def all_models_ok(loaded: dict) -> bool:
    return all(r.ok for r in loaded.values())


def get_estimator(pipeline_or_estimator):
    """Return the final estimator whether given a bare estimator or a Pipeline."""
    if hasattr(pipeline_or_estimator, "named_steps"):
        return pipeline_or_estimator.named_steps.get("model", list(pipeline_or_estimator.named_steps.values())[-1])
    return pipeline_or_estimator


def describe_model(pipeline_or_estimator) -> dict:
    """Introspect a loaded model for the About page's auto-generated metadata
    block - works for any sklearn-compatible estimator, not just the ones
    this project currently ships with."""
    estimator = get_estimator(pipeline_or_estimator)
    info = {
        "algorithm": type(estimator).__name__,
        "module": type(estimator).__module__,
    }
    try:
        params = estimator.get_params()
        # Keep this readable - drop overly verbose nested objects
        info["hyperparameters"] = {
            k: v for k, v in params.items()
            if not hasattr(v, "get_params")
        }
    except Exception:
        info["hyperparameters"] = {}

    if hasattr(estimator, "n_features_in_"):
        info["n_features_in"] = int(estimator.n_features_in_)
    if hasattr(estimator, "classes_"):
        info["classes"] = list(estimator.classes_)
    if hasattr(estimator, "feature_importances_"):
        info["has_feature_importances"] = True
    if hasattr(estimator, "coef_"):
        info["has_coefficients"] = True

    return info
