# Predicting the Severity of Political Violence in Kenya
---
A machine learning pipeline predicting the severity of political violence - measured as fatality counts - across Kenya's 47 counties at weekly resolution, using historical conflict data enriched with demographic features.  

**Team:** Ctrl-Alt-Elite  
**Contributors:** Kimutai Kevine, Mercy Wambui, Jackline Mwau, Richard Oketch, Winnie Nduva, Reeves Gonah  
**Date:** June - July 2026 (Capstone Project)  

---
## Elevator Pitch

Political violence places communities, election officials, humanitarian organisations, and security planners under significant pressure to make timely decisions with incomplete information. This project demonstrates how historical conflict patterns can be transformed into an early-warning tool that estimates where severe violence is most likely to occur.

Rather than attempting to replace expert judgement, the model provides a county-level risk assessment that helps prioritise monitoring, preparedness, and resource allocation before violence escalates.

---

## Overview
The project was designed around a practical question:

> Given what has happened historically in a county, can we estimate whether violence is likely to become fatal during the coming week, and if so, how severe it may be?

Answering this question required combining conflict event records with demographic information, engineering temporal features that capture recent patterns, and evaluating multiple machine learning approaches under realistic forecasting conditions.  

---
## Data Sources

| Source | Content | Coverage |
|---|---|---|
| [ACLED Africa Aggregated Data](https://acleddata.com/aggregated/aggregated-data-africa) | Conflict events and fatalities, filtered to Kenya (16,627 county-week rows) | 1997 – June 2026 |
| [WorldPop PWD](https://www.worldpop.org/) | County-level population, density, area, and population-weighted density | Census snapshots 2000–2020 |

WorldPop values are linearly interpolated between census snapshots, backfilled before 2000, and extrapolated after 2020 at 2.2% annual growth (KNBS 2019 inter-censal rate), then left-joined onto ACLED by county and year.

---

## Methodology

The pipeline follows CRISP-DM: business understanding → dataset creation → data understanding → cleaning → EDA → modelling → conclusions.

**Key data characteristics** (from EDA):
- Severely zero-inflated target: 72.7% of county-weeks record zero fatalities; skewness ~26; max 281 fatalities in one county-week
- Protests and riots dominate event counts but cause few deaths; Violence against civilians and Battles account for ~75% of fatalities
- Fatalities are ~25% higher per week in election years; deadliest counties are northern pastoral ones (Turkana, Mandera, Garissa)

**Modelling strategy — two-stage hurdle model:**
1. **Stage 1:** Binary classification - fatal vs non-fatal county-week (primary metric: F1, with Recall prioritised)
2. **Stage 2:** Severity classification of fatal weeks - Low (1-2), Medium (3-5), High (6+) fatalities (primary metric: Macro F1)

**Why these features?**  
- Political violence is rarely random. Previous conflict activity, election cycles, event characteristics, and demographic exposure all influence the likelihood that future incidents escalate. These relationships allow machine learning models to learn meaningful patterns without using future information, reducing the risk of target leakage.

Models compared in each stage: **Dummy baseline**, **Random Forest**, **Gradient Boosting**, **XGBoost**, and **TabNet**, plus a direct regression benchmark. Evaluation uses a *temporal split (train < 2022, test ≥ 2022)* to mimic real forecasting. Features include event type, county, calendar/election features, log-transformed population and density, lag and 4-week rolling features, with leakage controls documented in the notebook.

---

## Results

| Component | Best model | Key metrics (temporal test set) |
|---|---|---|
| Stage 1 (fatal vs non-fatal) | Random Forest | F1 64.3%, Recall 86.4% |
| Stage 2 (severity) | Random Forest | Macro F1 ~42% |
| Combined hurdle | Random Forest + Random Forest | Accuracy 75.2%, Macro F1 42.5% |

- TabNet achieved the highest Stage 1 recall (97.5%) but at low precision; Random Forest was retained for its better balance and interpretability.
- Direct regression underperformed the hurdle approach - only XGBoost achieved positive R² (0.07), confirming exact fatality counts are impractical to predict.
- SHAP analysis shows predictions are driven by event/sub-event type, event frequency, population exposure, and past fatalities.
- Predicted county risk aggregates into a vulnerability index and choropleth map; the top predicted counties (Mandera, Garissa, Turkana, Lamu, Isiolo, Samburu, Marsabit, Wajir) closely match the actual fatality hotspots.

---

## Limitations

- High-severity events are rare (0.5% of weeks); per-class F1 degrades from Non-fatal (0.87) to High (0.10) - the model ranks risk well but cannot yet reliably classify the rarest, most severe events.
- ACLED reporting coverage varies by county, period, and event visibility.
- The model learns historical patterns; sudden political shocks are not predictable. It supports risk prioritisation and does not replace expert judgement or field intelligence.

---

## Potential Stakeholders

Although developed as an academic capstone, the framework is applicable to organisations involved in conflict monitoring and disaster preparedness, including:

- Government agencies responsible for public safety and emergency planning.
- Election observers monitoring periods of elevated political tension.
- Humanitarian organisations prioritising deployment of limited resources.
- Researchers studying conflict dynamics within Kenya.
- Journalists and policy analysts tracking emerging hotspots.

The model is intended as a decision-support tool rather than a replacement for expert judgement.

A live interaction dashboard has been deployed via streamlit. To access: [follow this link](https://kenya-political-violence-risk-prediction.streamlit.app/)

---
## Repository Contents

- `master_notebook.ipynb` - full end-to-end pipeline jupyter notebook (dataset creation through modelling and mapping)
- `master_notebook_pdf.pdf` - pdf version of the jupyter notebook
- `presentation.pdf` - non-technical presentation powerpoint slides in pdf format.
- `Africa_aggregated_data.csv` - pre-filtered ACLED Kenya data (input)
- `PWD_2020-2000_sub_national_100m.csv` - WorldPop PWD data (input)
- `kenya_conflict_merged.csv` / `kenya_conflict_clean.csv` - generated intermediate datasets
- `kenya_conflict_dashboards` - generated html data dashboard page
- `Violence_image` - Header image,jupyter notebook
- `requirements.txt` - Requirements file to create environment to recreate run environment
- `kenya_counties_geojson` - File requierd to create kenya county choropleth map in notebook
- `.gitignore` - folders to intentionally ignore and leave untracked
- `kenya_app folder` - deployment files, inclusive of seperate dedicated readme section with file details
- `.pkl and .joblib files` - saved dataand model files for quick loading and deployment

---
## Reproducing the Project

This project was developed and tested using **Python 3.11**.

1. Clone the repository:

```bash
git clone https://github.com/KimutaiHub/kenya-political-violence-risk-prediction.git
cd kenya-political-violence-risk-prediction
```

2. Create a Python 3.11 virtual environment and activate it:

```bash
python3.11 -m venv .venv
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run `master_notebook.ipynb` to reproduce the complete analysis, or launch the deployment with:

```bash
cd kenya_app
streamlit run Home.py
```

The deployment includes an automated ACLED refresh pipeline (`deployment/pipeline/refresh_acled.py`) for updating the application with the latest available conflict data.

---
## References

This project was informed by literature and datasets from the conflict forecasting and humanitarian early warning domains.

1. ACLED (Armed Conflict Location & Event Data Project). https://acleddata.com/
2. Raleigh, C., Linke, A., Hegre, H., & Karlsen, J. (2010). *Introducing ACLED: An Armed Conflict Location and Event Dataset.*
3. WorldPop Project. https://www.worldpop.org/
4. Kenya National Bureau of Statistics (KNBS). 2019 Kenya Population and Housing Census.
