import streamlit as st

from utils.ui import inject_v2_styles, top_nav, intelligence_header, section_label, panel, disclaimer
from utils.data_loader import load_master_comparison, load_models
from utils.model_loader import describe_model

st.set_page_config(page_title="About the Model - Kenya Conflict Monitor", page_icon="ℹ", layout="wide")
inject_v2_styles()
top_nav(active_index=5)

comparison = load_master_comparison()
stage1_model, stage2_model, stage2_label_encoder, feature_lists, models_ok, model_errors = load_models()

intelligence_header("ABOUT THE MODEL", "Methodology, live model metadata, and known limitations")

st.markdown(
    """
    This tool is built on a **two-stage hurdle model**: Stage 1 predicts whether a county-week will
    have at least one conflict-related fatality; Stage 2, run only where Stage 1 predicts "yes,"
    classifies how severe that fatality count is likely to be (Low / Medium / High). The two-stage
    design exists because directly predicting fatality counts performed poorly - most county-weeks
    have zero fatalities, and a small number have extreme outliers.
    """
)

section_label("Live Model Metadata")
st.caption("Auto-generated from the models actually loaded in this deployment rather than a static description.")

meta_col1, meta_col2 = st.columns(2)

with meta_col1:
    with panel("Stage 1 - Fatal Event Detection"):
        if not models_ok and "stage1" in model_errors:
            st.error(model_errors.get("stage1", "Failed to load."))
        else:
            info = describe_model(stage1_model)
            st.markdown(f"**Algorithm:** `{info['algorithm']}`")
            if "n_features_in" in info:
                st.markdown(f"**Features in:** {info['n_features_in']}")
            if info.get("has_coefficients"):
                st.markdown("**Explainability:** Linear coefficients (SHAP LinearExplainer)")
            elif info.get("has_feature_importances"):
                st.markdown("**Explainability:** Tree feature importances (SHAP TreeExplainer)")
            with st.expander("Full hyperparameters"):
                st.json(info.get("hyperparameters", {}))

with meta_col2:
    with panel("Stage 2 - Severity Classification"):
        if not models_ok and "stage2" in model_errors:
            st.error(model_errors.get("stage2", "Failed to load."))
        else:
            info = describe_model(stage2_model)
            st.markdown(f"**Algorithm:** `{info['algorithm']}`")
            if "n_features_in" in info:
                st.markdown(f"**Features in:** {info['n_features_in']}")
            if "classes" in info:
                st.markdown(f"**Classes:** {', '.join(str(c) for c in info['classes'])}")
            with st.expander("Full hyperparameters"):
                st.json(info.get("hyperparameters", {}))

section_label("Model Performance")
with panel():
    st.dataframe(comparison, hide_index=True, use_container_width=True)

section_label("Limitations")
with panel():
    st.markdown(
        """
        - **Class imbalance limits severity granularity.** Performance degrades from Non-fatal
          (F1 = 0.85) to High severity (F1 = 0.11).
        - **Reporting bias in source data.** ACLED coverage depends on media and partner reporting,
          which varies by county and event visibility.
        - **Population features are estimates.** Post-2020 values are projected using a national
          growth rate distributed by each county's trending census share, not measured directly.
        - **Structural breaks are unpredictable.** The model learns historical patterns; sudden shocks
          can fall outside anything in the training data.
        - **County-week aggregation hides sub-county variation.** A high-risk ward inside a low-risk
          county is invisible at this resolution.
        - **SHAP explains the model, not reality.** Feature attributions describe how the model makes
          decisions, not causal evidence about what drives violence.
        """
    )

st.write("")
disclaimer(
    "This is a decision-support tool for risk prioritisation. It complements - and must never replace - "
    "expert judgement, field intelligence, and political context analysis."
)
