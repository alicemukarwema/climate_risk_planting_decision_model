"""
Small deployment smoke tests for the capstone API and Streamlit demo.

Run:  python tests/test_api.py

These tests avoid a running server and keep the checks close to the public
endpoint functions. They cover health, metrics, valid prediction, missing and
invalid inputs, required prediction fields, and a lightweight Streamlit import.
"""
import importlib
import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app


REQUIRED_FIELDS = {
    "crop",
    "planting_window",
    "risk_label",
    "class_probabilities",
    "p_rain_sufficient",
    "p_dry_spell",
    "p_temp_stress",
    "explanation",
}
VALID_LABELS = {"suitable", "risky", "delay"}
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))


def run_checks():
    app._startup()

    # 1. API health endpoint
    health = app.health()
    check(
        "health endpoint reports ready model",
        health["status"] == "ok"
        and health["artefacts_present"]
        and health["model_loaded"]
        and "Meteo Rwanda / ENACTS" in health["data_source"],
    )

    # 2. Metrics endpoint
    metrics = app.metrics()
    selected = metrics.get("selected_model_metrics", {})
    check(
        "metrics endpoint includes model comparison and selected metrics",
        {"rule_baseline", "dt_raw", "dt_risk", "xgb_full"} <= set(metrics)
        and bool(metrics.get("model_comparison_table"))
        and selected.get("macro_f1") is not None
        and selected.get("balanced_accuracy") is not None
        and selected.get("brier_score") is not None,
    )
    per_class = selected.get("per_class") or {}
    check(
        "metrics endpoint includes per-class recall",
        all("recall" in values for values in per_class.values()),
    )

    # 3. Valid prediction request
    valid = app.predict(app.Scenario(crop="maize", window_start_dekad=25))
    rec = valid["recommendation"]
    check(
        "valid prediction returns recommendation",
        rec["crop"] == "maize" and REQUIRED_FIELDS <= set(rec),
        f"label={rec.get('risk_label')}",
    )

    # 4. Missing and invalid input
    missing = app.predict(app.Scenario())
    missing_rec = missing["recommendation"]
    check(
        "missing input falls back to defaults",
        missing_rec["risk_label"] in VALID_LABELS,
        f"label={missing_rec.get('risk_label')}",
    )
    try:
        app.Scenario(crop="maize", window_start_dekad=50)
        invalid_rejected = False
    except ValidationError:
        invalid_rejected = True
    check("invalid planting window is rejected", invalid_rejected)

    # 5. Prediction output includes risk class and explanation
    check(
        "prediction includes risk class and explanation",
        rec["risk_label"] in VALID_LABELS and bool(rec["explanation"]),
    )
    check(
        "prediction includes risk components",
        all(k in rec for k in ("p_rain_sufficient", "p_dry_spell", "p_temp_stress")),
    )

    # 6. Streamlit import does not crash if dependency is installed
    try:
        streamlit_app = importlib.import_module("streamlit_app")
        report = streamlit_app.load_metrics()
        ok = report is None or "xgb_full" in report
        detail = "imported"
    except ModuleNotFoundError as exc:
        ok = False
        detail = f"missing dependency: {exc.name}"
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    check("streamlit_app imports", ok, detail)


run_checks()

print(f"\n{'TEST':58s} RESULT")
print("-" * 72)
ok = True
for name, passed, detail in results:
    ok &= passed
    print(
        f"{name:58s} {'PASS' if passed else 'FAIL'}"
        + (f"   [{detail}]" if detail else "")
    )
print("-" * 72)
print(f"{sum(p for _, p, _ in results)}/{len(results)} passed")
sys.exit(0 if ok else 1)
