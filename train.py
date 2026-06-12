"""
End-to-end training entrypoint (proposal pipeline, Figure 3):
data -> features -> stochastic simulation -> labels -> 4-model comparison
-> artefacts. Also exports the proposal's data tables (Table 4).
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from data_loader import load
from crops import CROPS, PLANTING_WINDOWS, DEKAD_LABEL
from features import yearly_features
from dataset import build, RAW_FEATURES, RISK_FEATURES
from model import train_final

if __name__ == "__main__":
    df = load()
    print(f"[1/4] data: {len(df)} dekads, {df.year.min()}-{df.year.max()}, "
          f"mean annual {df.groupby('year').rainfall_mm.sum().mean():.0f} mm")

    table, gen = build(df)
    print(f"[2/4] ML table: {len(table)} (year, window, crop) rows | "
          f"labels: {table.label.value_counts().to_dict()}")

    report = train_final(table)
    print("[3/4] model comparison (test 2015+):")
    for name in ("rule_baseline", "dt_raw", "dt_risk", "xgb_full"):
        m = report[name]
        print(f"   {name:14s} macro_f1={m['macro_f1']:.3f} "
              f"bal_acc={m['balanced_accuracy']:.3f} brier={m['brier_score']:.3f}")

    # ---- proposal Table 4 / capstone checklist exports ---------------
    data_dir = ROOT / "data"
    table.to_csv(data_dir / "ml_training_dataset.csv", index=False)
    yearly_features(df).to_csv(data_dir / "season_features.csv", index=False)
    pd.DataFrame([{"crop": k, **v} for k, v in CROPS.items()]) \
        .to_csv(data_dir / "crop_requirements.csv", index=False)
    pd.DataFrame([{"window_id": w, "season": "A",
                   "window_label": DEKAD_LABEL[w]} for w in PLANTING_WINDOWS]) \
        .to_csv(data_dir / "planting_windows.csv", index=False)

    # cleaned, merged climate table (dekadal equivalent of the checklist's
    # daily_climate_data.csv - the public ENACTS Maproom export is dekadal)
    clim = df.copy()
    clim["is_wet_dekad"] = (clim.rainfall_mm >= 20).astype(int)
    clim["season"] = clim.month.map(
        lambda m: "A (Sep-Dec)" if m >= 9 else
                  "B (Mar-May)" if 3 <= m <= 5 else "dry/other")
    clim["location"] = "Nyagatare box X 30.0-30.6, Y -1.5..-1.05 (spatial avg)"
    clim.to_csv(data_dir / "dekadal_climate_data.csv", index=False)

    # engineered features view (checklist's engineered_features.csv)
    from dataset import RAW_FEATURES, RISK_FEATURES
    table[["year", "crop"] + RAW_FEATURES + RISK_FEATURES + ["label"]] \
        .to_csv(data_dir / "engineered_features.csv", index=False)

    # Monte Carlo outcomes for EVERY crop x window (100 runs each)
    from crops import (DRY_DEKAD_MM, DRY_SPELL_DEKADS, ESTABLISHMENT_DEKADS)
    sims = []
    for crop, c in CROPS.items():
        for w in PLANTING_WINDOWS:
            rain, wet, tmax = gen.simulate(w, c["cycle_dekads"],
                                           n_sims=100, seed=w)
            for i in range(100):
                estab_wet = wet[i, :ESTABLISHMENT_DEKADS]
                run = worst = 0
                for is_wet in estab_wet:
                    run = 0 if is_wet else run + 1
                    worst = max(worst, run)
                total = float(rain[i].sum())
                estab_rain = float(rain[i, :ESTABLISHMENT_DEKADS].sum())
                sims.append({
                    "crop": crop, "window_id": w,
                    "window_label": DEKAD_LABEL[w], "simulation_run": i,
                    "sim_cycle_rain_mm": round(total, 1),
                    "sim_mean_tmax_c": round(float(tmax[i]), 2),
                    "rain_sufficient": int(total >= c["min_cycle_mm"]),
                    "estab_dry_spell": int(
                        worst >= DRY_SPELL_DEKADS
                        or estab_rain < c["min_establishment_mm"]),
                    "temp_stress": int(float(tmax[i]) > gen.tmax_clim_sond
                                       + c["tmax_stress_anom_c"]),
                })
    pd.DataFrame(sims).to_csv(data_dir / "simulation_outputs.csv", index=False)
    print("[4/4] artefacts -> models/ | tables -> data/ "
          "(dekadal_climate_data, engineered_features, ml_training_dataset, "
          "crop_requirements, planting_windows, simulation_outputs, "
          "season_features)")
