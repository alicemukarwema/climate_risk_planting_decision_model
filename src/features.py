"""
Feature engineering (proposal objective 2).

Two views of the data:

  yearly_features(df)   one row per year - EDA + season-level indicators
                        (onset dekad, Sep-Dec total, max dry run, ...)

  window_features(...)  one row per (year, planting window, crop) - the
                        observed *pre-window* conditions a decision-maker
                        would actually know at planting time (no leakage),
                        used as the "raw climate features" model input.
"""
import numpy as np
import pandas as pd

from crops import (CROPS, DRY_DEKAD_MM, ONSET_MM, DRY_SPELL_DEKADS,
                   ESTABLISHMENT_DEKADS, PLANTING_WINDOWS)
from data_loader import BASE_YEAR

SOND = (25, 36)
JUL_AUG = (19, 24)
MAM = (7, 15)


def _abs(year: int, dekad_of_year: int) -> int:
    return (year - BASE_YEAR) * 36 + dekad_of_year - 1


def yearly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Season-level indicators per year (EDA + API 'latest state')."""
    rows = []
    for year in sorted(df.year.unique()):
        sond = df[(df.year == year) & df.dekad_of_year.between(*SOND)]
        if len(sond) < 10:
            continue
        onset = next((int(r.dekad_of_year) for r in sond.itertuples()
                      if r.rainfall_mm >= ONSET_MM), SOND[1] + 1)
        dry = run = 0
        for r in sond.itertuples():
            run = run + 1 if r.rainfall_mm < DRY_DEKAD_MM else 0
            dry = max(dry, run)
        prev_mam = df[(df.year == year) & df.dekad_of_year.between(*MAM)]
        jul_aug = df[(df.year == year) & df.dekad_of_year.between(*JUL_AUG)]
        rows.append({
            "year": year,
            "onset_dekad": onset,
            "sond_total_mm": round(float(sond.rainfall_mm.sum()), 1),
            "max_dry_run": dry,
            "sond_tmax_mean": round(float(sond.tmax_c.mean()), 2)
                              if sond.tmax_c.notna().any() else np.nan,
            "pre_jul_aug_mm": round(float(jul_aug.rainfall_mm.sum()), 1),
            "mam_total_mm": round(float(prev_mam.rainfall_mm.sum()), 1),
        })
    out = pd.DataFrame(rows)
    out["prev_sond_total"] = out.sond_total_mm.shift(1)
    return out


def prewindow_features(rain: pd.Series, year: int, window: int,
                       clim_last3: tuple[float, float],
                       tmax: pd.Series | None = None,
                       clim_tmax_mj: float | None = None) -> dict | None:
    """Observed conditions known *before* planting dekad `window` of `year`.
    `rain` is the rainfall series indexed by abs_dekad;
    `clim_last3` = (mean, std) of 3-dekad rainfall ending at window-1,
    estimated from training years only;
    `tmax`/`clim_tmax_mj` give the May-Aug tmax anomaly of the current year
    (observable before Season A planting; captures warm years such as the
    2015/16 El Nino)."""
    w_abs = _abs(year, window)
    since_sep = rain.reindex(range(_abs(year, SOND[0]), w_abs))
    if since_sep.isna().any():
        return None
    last1 = rain.get(w_abs - 1, np.nan)
    last3 = rain.reindex(range(w_abs - 3, w_abs))
    if np.isnan(last1) or last3.isna().any():
        return None
    mean3, std3 = clim_last3
    onset_reached = int((since_sep >= ONSET_MM).any()) if len(since_sep) else 0

    pre_tmax_anom = 0.0          # 0 = climatological year (also when tmax missing)
    if tmax is not None and clim_tmax_mj is not None:
        mj = tmax.reindex(range(_abs(year, 13), _abs(year, 24) + 1))  # May-Aug
        if mj.notna().sum() >= 6:
            pre_tmax_anom = round(float(mj.mean() - clim_tmax_mj), 3)

    return {
        "window_start_dekad": window,
        "cum_rain_since_sep1": round(float(since_sep.sum()), 1),
        "onset_reached": onset_reached,
        "last_dekad_rain": round(float(last1), 1),
        "last3_rain": round(float(last3.sum()), 1),
        "last3_anom_z": round(float((last3.sum() - mean3) / max(std3, 1e-6)), 3),
        "prev_dekad_wet": int(last1 >= DRY_DEKAD_MM),
        "pre_jul_aug_mm": round(float(
            rain.reindex(range(_abs(year, JUL_AUG[0]),
                               _abs(year, JUL_AUG[1]) + 1)).sum()), 1),
        "prev_mam_mm": round(float(
            rain.reindex(range(_abs(year, MAM[0]),
                               _abs(year, MAM[1]) + 1)).sum()), 1),
        "prev_sond_total": round(float(
            rain.reindex(range(_abs(year - 1, SOND[0]),
                               _abs(year - 1, SOND[1]) + 1)).sum()), 1),
        "pre_tmax_anom": pre_tmax_anom,
    }


def observed_outcome(df_idx: pd.DataFrame, year: int, window: int,
                     crop: str, tmax_clim_sond: float) -> dict | None:
    """What actually happened after planting `crop` at `window` in `year`
    (used only to construct labels, never as a model input)."""
    c = CROPS[crop]
    w_abs = _abs(year, window)
    cycle = df_idx.reindex(range(w_abs, w_abs + c["cycle_dekads"]))
    if cycle.rainfall_mm.isna().any():
        return None
    rain_cycle = cycle.rainfall_mm.to_numpy()
    estab = rain_cycle[:ESTABLISHMENT_DEKADS]
    run = worst = 0
    for v in estab:
        run = run + 1 if v < DRY_DEKAD_MM else 0
        worst = max(worst, run)
    tmax_cycle = cycle.tmax_c.to_numpy(dtype=float)
    tmax_mean = float(np.nanmean(tmax_cycle)) if not np.isnan(tmax_cycle).all() \
        else tmax_clim_sond
    return {
        "actual_cycle_rain": round(float(rain_cycle.sum()), 1),
        "actual_estab_rain": round(float(estab.sum()), 1),
        "actual_estab_dry_run": int(worst),
        "actual_tmax_mean": round(tmax_mean, 2),
        "estab_dry_spell": int(worst >= DRY_SPELL_DEKADS
                               or estab.sum() < c["min_establishment_mm"]),
        "temp_stress": int(tmax_mean > tmax_clim_sond + c["tmax_stress_anom_c"]),
    }


if __name__ == "__main__":
    from data_loader import load
    f = yearly_features(load())
    print(f.tail().to_string(index=False))
    print(f.describe().round(1))
