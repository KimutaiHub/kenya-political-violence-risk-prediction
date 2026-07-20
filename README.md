# Predicting the Severity of Political Violence in Kenya

A machine learning pipeline predicting the severity of political violence — measured as fatality counts — across Kenya's 47 counties at weekly resolution, using historical conflict data enriched with demographic features.

**Team:** Ctrl-Alt-Elite
**Contributors:** Kimutai Kevine, Mercy Wambui, Jackline Mwau, Richard Oketch, Winnie Nduva, Reeves Gonah
**Date:** June – July 2026 (Capstone Project)

---

## Overview

Political violence in Kenya is patterned by geography, event type, population density, and electoral cycles — yet early warning and resource allocation still rely largely on expert judgement and lagged reporting. This project builds a data-driven forecasting pipeline that predicts, for each county-week, whether violence will turn fatal and how severe it will be.

## Data Sources

| Source | Content | Coverage |
|---|---|---|
| [ACLED Africa Aggregated Data](https://acleddata.com/aggregated/aggregated-data-africa) | Conflict events and fatalities, filtered to Kenya (16,627 county-week rows) | 1997 – June 2026 |
| [WorldPop PWD](https://www.worldpop.org/) | County-level population, density, area, and population-weighted density | Census snapshots 2000–2020 |

WorldPop values are linearly interpolated between census snapshots, backfilled before 2000, and extrapolated after 2020 at 2.2% annual growth (KNBS 2019 inter-censal rate), then left-joined onto ACLED by county and year.

## Methodology

The pipeline follows CRISP-DM: business understanding → dataset creation → data understanding → cleaning → EDA → modelling → conclusions.

**Key data characteristics** (from EDA):
- Severely zero-inflated target: 72.7% of county-weeks record zero fatalities; skewness ~26; max 281 fatalities in one county-week
- Protests and riots dominate event counts but cause few deaths; Violence against civilians and Battles account for ~75% of fatalities
- Fatalities are ~25% higher per week in election years; deadliest counties are northern pastoral ones (Turkana, Mandera, Garissa)

**Modelling strategy — two-stage hurdle model:**
1. **Stage 1:** Binary classification — fatal vs non-fatal county-week (primary metric: F1, with Recall prioritised)
2. **Stage 2:** Severity classification of fatal weeks — Low (1–2), Medium (3–5), High (6+) fatalities (primary metric: Macro F1)

Models compared in each stage: Dummy baseline, Random Forest, Gradient Boosting, XGBoost, and TabNet, plus a direct regression benchmark. Evaluation uses a temporal split (train < 2022, test ≥ 2022) to mimic real forecasting. Features include event type, county, calendar/election features, log-transformed population and density, lag and 4-week rolling features, with leakage controls documented in the notebook.

## Results

| Component | Best model | Key metrics (temporal test set) |
|---|---|---|
| Stage 1 (fatal vs non-fatal) | Random Forest | F1 64.7%, Recall 88.9% |
| Stage 2 (severity) | Random Forest | Macro F1 ~41% |
| Combined hurdle | Random Forest + Random Forest | Accuracy 75.2%, Macro F1 42.1% |

- TabNet achieved the highest Stage 1 recall (95.8%) but at low precision; Random Forest was retained for its better balance and interpretability.
- Direct regression underperformed the hurdle approach — only XGBoost achieved positive R² (0.07), confirming exact fatality counts are impractical to predict.
- SHAP analysis shows predictions are driven by event/sub-event type, event frequency, population exposure, and past fatalities.
- Predicted county risk aggregates into a vulnerability index and choropleth map; the top predicted counties (Mandera, Garissa, Turkana, Lamu, Isiolo, Samburu, Marsabit, Wajir) closely match the actual fatality hotspots.

## Limitations

- High-severity events are rare (0.5% of weeks); per-class F1 degrades from Non-fatal (0.87) to High (0.10) — the model ranks risk well but cannot yet reliably classify the rarest, most severe events.
- ACLED reporting coverage varies by county, period, and event visibility.
- The model learns historical patterns; sudden political shocks are not predictable. It supports risk prioritisation and does not replace expert judgement or field intelligence.

## Repository Contents

- `master_notebook_revised_final.ipynb` — full end-to-end pipeline (dataset creation through modelling and mapping)
- `Africa_aggregated_data.csv` — pre-filtered ACLED Kenya data (input)
- `PWD_2020-2000_sub_national_100m.csv` — WorldPop PWD data (input)
- `kenya_conflict_merged.csv` / `kenya_conflict_clean.csv` — generated intermediate datasets

## Requirements

Python 3 with: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `pytorch-tabnet`, `torch`, `shap`, `squarify`, `missingno`, `scipy`, `requests`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost pytorch-tabnet torch shap squarify missingno scipy requests
```
## Authors

Team Ctrl-Alt-Elite — Moringa School Data Science Capstone

- Kimutai Kevine
- Mercy Wambui
- Jackline Mwau
- Richard Oketch
- Winnie Nduva 
- Reeves Gonah

