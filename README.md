# Climate Risk-Aware Planting Window Classification Model for Maize and Beans in Nyagatare District

This capstone project classifies maize and bean planting windows in Nyagatare
District as **suitable**, **risky**, or **delay** using Meteo Rwanda/ENACTS
dekadal climate data and a trained machine-learning model.

> **GitHub repo:** https://github.com/alicemukarwema/climate_risk_planting_decision_model  
> **Demo video:** https://drive.google.com/file/d/1trDtNwObJ4aEENOfXE2WRl2Xh8jfeOKN/view?usp=sharing

## What The Project Does

The project answers one focused question:

```text
For a selected crop and planting window in Nyagatare, is the climate risk
suitable, risky, or high enough that planting should be delayed?
```

The current implementation supports:

- crops: maize and beans
- location: Nyagatare District / Nyagatare area
- season: Season A planting windows from September to November
- outputs: risk class, class probabilities, risk components, and a short explanation

The model is a decision-support prototype for academic demonstration and
supervisor testing. It is not a farmer-facing production advisory system.

## Dataset

The dataset uses **Meteo Rwanda/ENACTS Maproom dekadal climate extracts for
Nyagatare**. The exact CSV files used in this project are committed in the
`data/` folder:
- `data/nyagatare_rainfall_dekadal.csv`
- `data/nyagatare_tmax.csv`
- `data/nyagatare_tmin.csv`

The data are dekadal, meaning one raw row represents one 10-day climate period.
They are grid/area-average climate extracts for the Nyagatare area, not
farm-level measurements.

Dataset documentation: [docs/DATASET.md](docs/DATASET.md)

## ML Task

This is a **multi-class classification** task.

Each modelling row represents:

```text
one year x one crop x one candidate planting window
```

The target classes are:

- `suitable`
- `risky`
- `delay`

The labels are proxy agronomic risk labels. They are created by comparing
climate outcomes after each candidate planting window with maize and bean
rainfall, dry-spell, and temperature thresholds. They are not measured yield
labels.

Label definition: [docs/label-definition/README.md](docs/label-definition/README.md)

## Models Compared

The project compares three model families:

- rule-based baseline
- Decision Tree models
- XGBoost multi-class classifier

The selected model is the XGBoost classifier using raw climate features and
stochastic risk features.

Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

## Metrics

The main evaluation uses a temporal hold-out split:

- training years: 1982-2014
- test years: 2015-2023

The project reports:

- macro F1
- balanced accuracy
- Brier score
- confusion matrix
- per-class recall

Initial model comparison:

| model | macro F1 | balanced accuracy | Brier score |
|---|---:|---:|---:|
| Rule-based baseline | 0.320 | 0.340 | 1.068 |
| Decision Tree - raw climate | 0.463 | 0.542 | 0.923 |
| Decision Tree - stochastic risk | 0.526 | 0.632 | 0.937 |
| XGBoost - all features | 0.642 | 0.706 | 0.556 |

Selected model per-class recall:

| class | recall |
|---|---:|
| suitable | 0.474 |
| risky | 0.866 |
| delay | 0.778 |

The full report is saved in `models/report.json`.

## Demo

### Local Streamlit Demo

```bash
streamlit run streamlit_app.py
```

### Local FastAPI Demo

```bash
uvicorn app:app --reload --port 8000
```

Then open:

| URL | Purpose |
|---|---|
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/health | API and model status |
| http://localhost:8000/predict | Prediction endpoint |
| http://localhost:8000/metrics | Model metrics endpoint |

## Deployment Links

- Streamlit demo: https://climateriskplantingdecisionmodel-mqfkjhaejf5e9fk5q8slgt.streamlit.app/
- FastAPI demo: **TO BE ADDED IF DEPLOYED**

## Setup

Requires **Python 3.11+**.

```bash
git clone https://github.com/alicemukarwema/climate_risk_planting_decision_model
cd climate_risk_planting_decision_model
```

Create and activate a virtual environment.

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train or refresh the model artifacts:

```bash
python train.py
```

Run the capstone smoke tests:

```bash
python tests/test_api.py
```

## Project Structure

```text
app.py                         FastAPI app
streamlit_app.py               Streamlit demo interface
train.py                       Training entry point
src/                           Data loading, features, simulation, modelling, recommendations
data/                          Meteo Rwanda/ENACTS extracts and generated CSV tables
models/                        Saved XGBoost model and metrics report
notebooks/nyagatare_model.ipynb Executed modelling notebook
tests/test_api.py              Capstone smoke tests
docs/                          Dataset, label, model card, figures, and screenshots
```

## Limitations

- The model does not predict yield.
- The model does not use farm-level measurements.
- The output is decision support only, not guaranteed farming advice.
- Crop thresholds are proxy agronomic thresholds and require agronomist/RAB validation before farmer-facing use.
- The current implementation does not include a mobile app, SMS, USSD, or dashboard.

## Future Work

- Agronomist/RAB validation of crop thresholds and label rules.
- Wider farmer and extension-officer testing in Nyagatare.
- Mobile app interface.
- SMS/USSD access for low-connectivity users.
- Dashboard for monitoring model outputs and feedback.
- Automatic retraining when updated climate records are added.
- Use of daily or sector-level climate data if accessible.
