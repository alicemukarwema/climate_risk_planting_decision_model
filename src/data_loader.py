"""
Data layer: parse the real Meteo Rwanda ENACTS dekadal exports.

Files (exported from Meteo Rwanda ENACTS Maproom/Data services; access through
https://www.meteorwanda.gov.rw/home, Nyagatare box X 30.0..30.6,
Y -1.5..-1.05, spatial [X Y] average, Data Files page):

  data/nyagatare_rainfall_dekadal.csv  Time, Merged Station-Satellite Rainfall   1981-2023
  data/nyagatare_tmax.csv              Time, Gridded Maximum Temperature         1961-2021
  data/nyagatare_tmin.csv              Time, Gridded Minimum Temperature         1961-2016

The ENACTS "Time" column labels each dekad as e.g. "1-10 Jan 1981",
"11-20 Jan 1981", "21-31 Jan 1981" (third dekad runs to month end).
Missing values (-99) are converted to NaN and dropped.

Output of load(): one row per dekad with
  date, year, month, dekad_of_year (1..36), abs_dekad (continuous index),
  rainfall_mm, tmax_c, tmin_c
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
BASE_YEAR = 1961          # for the continuous dekad index

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _parse_enacts(path: Path, value_name: str) -> pd.DataFrame:
    """Parse one ENACTS dekadal CSV into (year, month, dekad_in_month, value)."""
    df = pd.read_csv(path)
    df.columns = ["time", value_name]
    parts = df["time"].str.strip().str.split(" ", expand=True)
    day0 = parts[0].str.split("-", expand=True)[0].astype(int)
    df["dekad_in_month"] = day0.map({1: 1, 11: 2, 21: 3})
    df["month"] = parts[1].map(_MONTHS)
    df["year"] = parts[2].astype(int)
    df[value_name] = df[value_name].replace(-99, np.nan)
    return df[["year", "month", "dekad_in_month", value_name]]


def load() -> pd.DataFrame:
    """Merged dekadal climate table for Nyagatare (rainfall + tmax + tmin)."""
    rain = _parse_enacts(DATA / "nyagatare_rainfall_dekadal.csv", "rainfall_mm")
    tmax = _parse_enacts(DATA / "nyagatare_tmax.csv", "tmax_c")
    tmin = _parse_enacts(DATA / "nyagatare_tmin.csv", "tmin_c")

    keys = ["year", "month", "dekad_in_month"]
    df = rain.merge(tmax, on=keys, how="left").merge(tmin, on=keys, how="left")
    df = df.dropna(subset=["rainfall_mm"]).reset_index(drop=True)

    df["dekad_of_year"] = (df["month"] - 1) * 3 + df["dekad_in_month"]
    df["abs_dekad"] = (df["year"] - BASE_YEAR) * 36 + df["dekad_of_year"] - 1
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month,
                                     day=(df.dekad_in_month - 1) * 10 + 1))
    df = df.sort_values("abs_dekad").reset_index(drop=True)
    return df[["date", "year", "month", "dekad_of_year", "abs_dekad",
               "rainfall_mm", "tmax_c", "tmin_c"]]


def temperature_pools(df: pd.DataFrame, year_max: int | None = None) -> dict:
    """Empirical tmax values per dekad_of_year (gridded 1961-2021 record)
    for stochastic temperature sampling. `year_max` limits the pool to
    training years (leakage control during evaluation)."""
    t = _parse_enacts(DATA / "nyagatare_tmax.csv", "tmax_c").dropna()
    if year_max is not None:
        t = t[t.year <= year_max]
    t["dekad_of_year"] = (t["month"] - 1) * 3 + t["dekad_in_month"]
    return {d: g["tmax_c"].to_numpy() for d, g in t.groupby("dekad_of_year")}


if __name__ == "__main__":
    d = load()
    print(d.head(3).to_string(index=False))
    print(f"\n{len(d)} dekads | {d.year.min()}-{d.year.max()} | "
          f"mean annual rain = {d.groupby('year').rainfall_mm.sum().mean():.0f} mm | "
          f"tmax coverage to {d.dropna(subset=['tmax_c']).year.max()}")
