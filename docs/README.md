# Dataset Used in This Project

This project uses **Meteo Rwanda / ENACTS Maproom dekadal climate extracts for
Nyagatare**.

The data source is the **Rwanda Meteorology Agency / Meteo Rwanda ENACTS
Maproom/Data portal**. The portal provides public climate data exports, but the
Maproom pages may require manual navigation when downloading. To make the
project reproducible, the exact CSV files used here are already committed in
the `data/` folder.

## Raw Files

The model uses three climate files:

| file | what it contains |
|---|---|
| `data/nyagatare_rainfall_dekadal.csv` | Dekadal rainfall totals for 1981-2023 |
| `data/nyagatare_tmax.csv` | Dekadal maximum temperature for 1961-2021 |
| `data/nyagatare_tmin.csv` | Dekadal minimum temperature for 1961-2016 |

## What the Data Represents

The geographic scope is **Nyagatare District / the Nyagatare area** in Rwanda.
These records are ENACTS grid or area-average climate extracts for the selected
Nyagatare area. They are not farm-level measurements.

This means the data are useful for district-level climate risk modelling, but
they should not be read as the exact rainfall or temperature experienced on a
specific farm.

## Unit of Analysis

In the raw climate files, one row represents one **dekad**, which is a 10-day
climate period. A month has three dekads: days 1-10, days 11-20, and days 21 to
the end of the month.

For modelling, the data are transformed into a planting-risk table where one row
represents:

```text
one year x one crop x one candidate planting window
```

Each modelling row combines the climate conditions before planting, stochastic
risk estimates, observed crop-window outcomes, and the final target label.

## Why the Project Uses Dekadal Data

The publicly accessible ENACTS export used for this project is dekadal, so the
pipeline works with 10-day rainfall and temperature periods instead of daily
records.

Daily data would be a useful future improvement. It would allow the model to
capture shorter dry spells and short heat events more directly.

## Variables Used

The project starts from rainfall, maximum temperature, and minimum temperature.
From these, the pipeline builds additional features used by the recommendation
model, including:

- derived mean temperature
- cumulative rainfall
- rainfall onset indicators
- rainfall anomaly
- dry-spell risk
- rainfall sufficiency probability
- temperature stress probability

Rainfall is the most important variable for planting-window risk because it is
used to estimate onset, dry spells, crop-cycle rainfall, and rainfall
sufficiency.

## Missing Values

The ENACTS CSV files use `-99` to mark missing values. During loading, those
values are converted to `NaN`.

Rows with missing rainfall are removed from the merged climate table because
rainfall is required for the main planting-risk calculations.

Temperature coverage is shorter than rainfall coverage. Where temperature values
are unavailable, the modelling pipeline uses climatology-based fallback values
for temperature anomaly and temperature stress calculations.

## Target Label

The model predicts one of three planting-risk labels:

- `suitable`: conditions meet the crop rainfall requirement and do not show
  harmful temperature stress
- `risky`: conditions are not clearly suitable, but are also not severe enough
  to recommend delay
- `delay`: conditions are too dry, the establishment phase fails, or expected
  crop-cycle rainfall is far below the crop requirement

## Limitations

This dataset is good enough for an MVP, but it has important limits:

- Dekadal data cannot capture very short dry spells inside a 10-day period.
- Spatial averages are not the same as farm-level measurements.
- Crop thresholds are proxy thresholds and need agronomist/RAB validation
  before real farmer use.
- The model output is decision support only, not a guaranteed outcome.
