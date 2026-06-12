"""
Advisory service: glue between the trained artefacts and the API.

At startup it fits the weather generator on the full historical record,
loads the selected XGBoost model, and precomputes climatological defaults
per planting window so a caller can ask "what if?" with as little or as
much detail as they like (proposal objective 6).
"""
import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from crops import CROPS, PLANTING_WINDOWS, LABELS
from data_loader import load, temperature_pools
from dataset import RAW_FEATURES, RISK_FEATURES, _clim_last3
from features import _abs
from model import load_final, MODELS
from recommend import package, pick_best
from simulate import WeatherGenerator

LOGS = Path(__file__).resolve().parent.parent / "data" / "prediction_logs.csv"
ALL_FEATURES = RAW_FEATURES + RISK_FEATURES


class AdvisoryService:
    def __init__(self):
        self.df = load()
        self.gen = WeatherGenerator().fit(self.df, temperature_pools(self.df))
        self.model = load_final()
        rain = self.df.set_index("abs_dekad")["rainfall_mm"]
        years = range(int(self.df.year.min()) + 1, int(self.df.year.max()) + 1)
        self.clim3 = {w: _clim_last3(rain, w, years) for w in PLANTING_WINDOWS}

        # climatological defaults for the raw features, per window
        self.defaults = {}
        for w in PLANTING_WINDOWS:
            cum, last1 = [], []
            for y in years:
                w_abs = _abs(y, w)
                seg = rain.reindex(range(_abs(y, 25), w_abs))
                if not seg.isna().any():
                    cum.append(float(seg.sum()))
                v = rain.get(w_abs - 1, np.nan)
                if not np.isnan(v):
                    last1.append(float(v))
            self.defaults[w] = {
                "cum_rain_since_sep1": round(float(np.mean(cum)) if cum else 0.0, 1),
                "last_dekad_rain": round(float(np.mean(last1)), 1),
                "last3_rain": round(self.clim3[w][0], 1),
            }
        ja = self.df[self.df.dekad_of_year.between(19, 24)]
        mam = self.df[self.df.dekad_of_year.between(7, 15)]
        sond = self.df[self.df.dekad_of_year.between(25, 36)]
        self.global_defaults = {
            "pre_jul_aug_mm": round(float(ja.groupby("year").rainfall_mm.sum().mean()), 1),
            "prev_mam_mm": round(float(mam.groupby("year").rainfall_mm.sum().mean()), 1),
            "prev_sond_total": round(float(sond.groupby("year").rainfall_mm.sum().mean()), 1),
            "pre_tmax_anom": 0.0,
        }

    # ------------------------------------------------------------------
    def predict_option(self, crop: str, window: int,
                       overrides: dict | None = None) -> dict:
        """Risk-aware assessment of one (crop, window) option."""
        overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        d = {**self.defaults[window], **self.global_defaults, **overrides}
        mean3, std3 = self.clim3[window]
        last3 = float(d["last3_rain"])
        feats = {
            "window_start_dekad": window,
            "crop_is_maize": int(crop == "maize"),
            "cum_rain_since_sep1": float(d["cum_rain_since_sep1"]),
            "onset_reached": int(d.get(
                "onset_reached", d["cum_rain_since_sep1"] >= 25 or window == 25)),
            "last_dekad_rain": float(d["last_dekad_rain"]),
            "last3_rain": last3,
            "last3_anom_z": round((last3 - mean3) / max(std3, 1e-6), 3),
            "prev_dekad_wet": int(float(d["last_dekad_rain"]) >= 20),
            "pre_jul_aug_mm": float(d["pre_jul_aug_mm"]),
            "prev_mam_mm": float(d["prev_mam_mm"]),
            "prev_sond_total": float(d["prev_sond_total"]),
            "pre_tmax_anom": float(d["pre_tmax_anom"]),
        }
        risk = self.gen.risk_features(
            crop, window, init_wet=bool(feats["prev_dekad_wet"]),
            anom_z=feats["last3_anom_z"], tmax_anom=feats["pre_tmax_anom"],
            seed=window)
        row = pd.DataFrame([{**feats, **{k: risk[k] for k in RISK_FEATURES}}])
        proba = self.model.predict_proba(row[ALL_FEATURES])[0]
        option = package(crop, window, proba, risk)
        option["inputs_used"] = feats
        return option

    def season_scan(self, overrides: dict | None = None) -> dict:
        """Assess every (crop, window) option and rank them."""
        options = [self.predict_option(crop, w, overrides)
                   for crop in CROPS for w in PLANTING_WINDOWS]
        result = pick_best(options)
        result["all_options"] = sorted(options,
                                       key=lambda o: (o["crop"],
                                                      o["window_start_dekad"]))
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def log_prediction(option: dict):
        """Append to prediction_logs.csv (proposal Table 4)."""
        LOGS.parent.mkdir(exist_ok=True)
        new = not LOGS.exists()
        with LOGS.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "crop", "window", "risk_label",
                            "risk_score", "confidence", "explanation"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"),
                        option["crop"], option["planting_window"],
                        option["risk_label"], option["risk_score"],
                        option["confidence"], option["explanation"]])

    @staticmethod
    def artefacts_present() -> bool:
        return (MODELS / "report.json").exists() and \
               (MODELS / "xgb_planting_risk.json").exists()
