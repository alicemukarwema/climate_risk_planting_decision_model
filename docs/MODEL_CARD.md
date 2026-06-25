# Model Card

## Model Name

**Climate Risk-Aware Planting Window Classifier**

## Model Type

The selected model is an **XGBoost multi-class classifier**. It was selected
after comparison with a rule-based baseline and Decision Tree models.

The task is to classify planting-window risk into three classes:

- `suitable`
- `risky`
- `delay`

## Intended Use

This model is intended for an academic capstone demonstration and supervisor
testing of maize and bean planting-window risk classification for the Nyagatare
area.

It supports the project goal of showing how historical climate records can be
turned into a risk-aware planting-window recommendation workflow.

## Not Intended For

This model is not intended for:

- direct farmer use without agronomist validation
- national deployment
- yield prediction
- profit prediction
- guaranteed planting decisions

It should not be presented as a replacement for Meteo Rwanda, RAB, or
agronomists.

## Input Data

The model uses engineered dekadal rainfall and temperature features from the
Meteo Rwanda / ENACTS Nyagatare climate extracts.

Input features include pre-window climate indicators such as recent rainfall,
cumulative rainfall, rainfall onset, rainfall anomaly, previous-season rainfall,
and temperature anomaly. The final model also uses stochastic risk features such
as rainfall sufficiency probability, dry-spell probability, temperature stress
probability, and risk score.

## Output

The deployed service returns:

- predicted risk class
- class probabilities
- recommended crop and planting window
- stochastic risk components
- a short plain-language explanation

The output is meant to help compare crop-window options, not to guarantee a
specific farming result.

## Evaluation

The main evaluation uses a temporal hold-out split:

- training years: 1982-2014
- test years: 2015-2023
- test rows: 133

The selected XGBoost model achieved:

| metric | value |
|---|---:|
| Macro F1 | 0.642 |
| Balanced accuracy | 0.706 |
| Brier score | 0.556 |

Per-class recall on the test set:

| class | recall | support |
|---|---:|---:|
| `suitable` | 0.474 | 57 |
| `risky` | 0.866 | 67 |
| `delay` | 0.778 | 9 |

Confusion matrix for the selected model:

Rows are true labels. Columns are predicted labels.

| true \ predicted | `suitable` | `risky` | `delay` |
|---|---:|---:|---:|
| `suitable` | 27 | 29 | 1 |
| `risky` | 0 | 58 | 9 |
| `delay` | 0 | 2 | 7 |

The comparison report is saved in `models/report.json`, and notebook figures
include confusion matrices and model-comparison plots.

## Most Important Error

The most important error is **false reassurance**: predicting `suitable` when
the real condition is actually `delay` or otherwise high risk.

This error matters more than an overly cautious prediction because it could
encourage planting during a poor window. In the current temporal test set, the
selected model did not predict `suitable` for any true `delay` cases, but false
reassurance remains the key safety risk to monitor before any real-world use.

## Limitations

- Dekadal climate data are not farm-level measurements.
- Dekadal data can miss short dry spells inside a 10-day period.
- Temperature data periods may be shorter than rainfall coverage.
- Crop thresholds are proxy thresholds, not locally validated yield labels.
- The model needs agronomist/RAB validation before farmer-facing use.

## Ethical Statement

The output is decision support only and should not be presented as guaranteed
farming advice. Any farmer-facing version would need agronomist validation,
clear uncertainty communication, and review by relevant local institutions.
