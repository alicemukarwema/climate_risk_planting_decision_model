# Label Definition

This project treats planting-window recommendation as a **multi-class
classification of planting-window risk**.

For each candidate planting window, the model estimates whether planting a
specific crop in the Nyagatare area is likely to be suitable, risky, or better
delayed.

## Target Classes

The model uses three target classes:

| class | meaning |
|---|---|
| `suitable` | The climate outcome after planting meets the crop rainfall requirement and does not show harmful temperature stress. |
| `risky` | The climate outcome is between clearly suitable and clearly delayed. |
| `delay` | The climate outcome is too dry, the establishment phase fails, or crop-cycle rainfall is far below the crop requirement. |

## Unit of Prediction

One prediction represents:

```text
one crop in one candidate planting window for the Nyagatare area
```

For example, the model can score maize planted in one September dekad and beans
planted in the same dekad as two separate crop-window cases.

## Where the Labels Come From

The labels are **proxy agronomic risk labels**, not measured yield labels.

They are created by comparing the climate outcomes after each candidate
planting window with maize and bean thresholds for:

- crop-cycle rainfall
- rainfall during the establishment phase
- harmful dry spells
- temperature stress

In the project code, a crop-window case is labelled `delay` when crop-cycle
rainfall is far below the crop minimum or the establishment phase fails. It is
labelled `suitable` when crop-cycle rainfall meets the crop minimum and there is
no temperature stress. Cases between those two conditions are labelled `risky`.

These rules make the labels transparent and reproducible, but they should still
be treated as agronomic proxies.

## Leakage Control

The model should only use information that would be available at prediction
time. For planting-window advice, that means the input features should describe
conditions known before the candidate planting window, not climate outcomes that
happen after planting.

To reduce leakage:

- pre-window climate features are used as model inputs
- temporal train/test splits are preferred over random splits where possible
- stochastic features should be fitted using training years where possible
- future climate outcomes are used to create labels for evaluation, not as
  prediction-time inputs

This is important because random splits can make time-dependent climate models
look better than they really are.

## What the Model Does Not Predict

The model does not predict yield.

It does not predict profit.

It does not guarantee farmer outcomes.

It does not replace Meteo Rwanda, RAB, or agronomists.

The output should be read as decision support: a structured way to compare crop
and planting-window risk using the available climate records.

## Why This Label Is Acceptable for the MVP

Measured yield labels were not available for every crop, year, and planting
window in this project. The proxy label allows the project to test whether
climate records can be transformed into a defensible planting-window risk
classification model.

Because the label is based on documented rainfall, dry-spell, and temperature
thresholds, it is explainable and can be reviewed by supervisors, agronomists,
or RAB experts before any real farmer-facing use.
