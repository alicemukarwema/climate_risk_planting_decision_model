"""
ML training dataset builder (proposal Table 4: ml_training_dataset).

One row per (year, candidate planting window, crop) combining:
  - raw pre-window climate features (what a decision-maker observes)
  - stochastic risk features (Markov chain + Monte Carlo, fitted on
    training years only to avoid leakage)
  - the suitability label derived from the *observed* outcome of that
    window against documented crop requirements (proposal objective 4):

      delay     cycle rain < DELAY_FACTOR x crop minimum, or the
                establishment phase failed (harmful dry spell / too dry)
      suitable  cycle rain >= crop minimum, establishment OK, no
                temperature stress
      risky     everything in between
"""
import numpy as np
import pandas as pd

from crops import CROPS, PLANTING_WINDOWS, DELAY_FACTOR, LABELS
from data_loader import BASE_YEAR, load, temperature_pools
from features import prewindow_features, observed_outcome, _abs
from simulate import WeatherGenerator

RAW_FEATURES = ["window_start_dekad", "crop_is_maize", "cum_rain_since_sep1",
                "onset_reached", "last_dekad_rain", "last3_rain",
                "last3_anom_z", "prev_dekad_wet", "pre_jul_aug_mm",
                "prev_mam_mm", "prev_sond_total", "pre_tmax_anom"]
RISK_FEATURES = ["p_rain_sufficient", "p_dry_spell", "p_temp_stress",
                 "risk_score", "sim_margin_median", "sim_margin_p10"]
SIM_EXTRA = ["sim_cycle_rain_p10", "sim_cycle_rain_median",
             "sim_cycle_rain_p90"]


def label_outcome(crop: str, outcome: dict) -> str:
    c = CROPS[crop]
    if (outcome["actual_cycle_rain"] < DELAY_FACTOR * c["min_cycle_mm"]
            or outcome["estab_dry_spell"]):
        return "delay"
    if (outcome["actual_cycle_rain"] >= c["min_cycle_mm"]
            and not outcome["temp_stress"]):
        return "suitable"
    return "risky"


def _clim_last3(rain: pd.Series, window: int, train_years: range) -> tuple:
    """Climatological mean/std of the 3 dekads preceding `window`."""
    vals = []
    for y in train_years:
        w_abs = _abs(y, window)
        v = rain.reindex(range(w_abs - 3, w_abs))
        if not v.isna().any():
            vals.append(v.sum())
    return float(np.mean(vals)), float(np.std(vals))


def build(df: pd.DataFrame | None = None, train_year_max: int = 2014,
          n_sims: int = 1000) -> tuple[pd.DataFrame, WeatherGenerator]:
    """Build the full ML table. Stochastic features and climatologies are
    fitted on years <= train_year_max only."""
    if df is None:
        df = load()
    rain = df.set_index("abs_dekad")["rainfall_mm"]
    tmax = df.set_index("abs_dekad")["tmax_c"]
    df_idx = df.set_index("abs_dekad")[["rainfall_mm", "tmax_c"]]
    gen = WeatherGenerator().fit(df, temperature_pools(df, train_year_max),
                                 year_max=train_year_max)

    train_years = range(int(df.year.min()) + 1, train_year_max + 1)
    clim3 = {w: _clim_last3(rain, w, train_years) for w in PLANTING_WINDOWS}
    # May-Aug tmax climatology (training years) for the pre-season anomaly
    mj = df[(df.dekad_of_year.between(13, 24)) & (df.year <= train_year_max)]
    clim_tmax_mj = float(mj.tmax_c.mean())

    rows = []
    for year in sorted(df.year.unique())[1:]:        # skip first (needs lags)
        for window in PLANTING_WINDOWS:
            pre = prewindow_features(rain, year, window, clim3[window],
                                     tmax, clim_tmax_mj)
            if pre is None:
                continue
            for crop in CROPS:
                out = observed_outcome(df_idx, year, window, crop,
                                       gen.tmax_clim_sond)
                if out is None:
                    continue
                risk = gen.risk_features(
                    crop, window, init_wet=bool(pre["prev_dekad_wet"]),
                    anom_z=pre["last3_anom_z"],
                    tmax_anom=pre["pre_tmax_anom"], n_sims=n_sims,
                    seed=year * 100 + window)
                rows.append({
                    "year": year, "crop": crop,
                    "crop_is_maize": int(crop == "maize"),
                    **pre,
                    **{k: risk[k] for k in RISK_FEATURES + SIM_EXTRA},
                    **out,
                    "label": label_outcome(crop, out),
                })
    table = pd.DataFrame(rows)
    table["label_id"] = table.label.map({l: i for i, l in enumerate(LABELS)})
    return table, gen


if __name__ == "__main__":
    table, _ = build(n_sims=400)
    print(f"{len(table)} rows | years {table.year.min()}-{table.year.max()}")
    print("\nlabel balance:\n", table.label.value_counts(normalize=True).round(3))
    print("\nby crop:\n", table.groupby('crop').label.value_counts(normalize=True).round(2))
    print("\nrisk_score by label:\n",
          table.groupby("label").risk_score.describe().round(3))
