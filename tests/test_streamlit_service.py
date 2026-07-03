"""
Smoke tests for the Streamlit capstone demo and its model service.

Run:
    .venv/bin/python tests/test_streamlit_service.py

These tests avoid a running browser/server and check the same service layer
used by streamlit_app.py: model artefacts, metrics, valid predictions,
season-wide recommendations, supported input choices, and Streamlit import.
"""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from crops import CROPS, PLANTING_WINDOWS
from model import MODELS
from service import AdvisoryService


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
    # 1. Model artefact readiness
    check(
        "model artefacts are present",
        AdvisoryService.artefacts_present(),
    )
    service = AdvisoryService()

    # 2. Metrics report
    report_path = MODELS / "report.json"
    report = json.loads(report_path.read_text())
    selected = report.get("xgb_full", {})
    check(
        "metrics report includes model comparison and selected model",
        {"rule_baseline", "dt_raw", "dt_risk", "xgb_full"} <= set(report)
        and selected.get("macro_f1") is not None
        and selected.get("balanced_accuracy") is not None
        and selected.get("brier_score") is not None,
    )
    per_class = selected.get("per_class") or {}
    check(
        "metrics report includes per-class recall",
        all("recall" in values for values in per_class.values()),
    )

    # 3. Valid Streamlit form values
    rec = service.predict_option("maize", 25)
    check(
        "valid form values return a recommendation",
        rec["crop"] == "maize"
        and rec["risk_label"] in VALID_LABELS
        and REQUIRED_FIELDS <= set(rec),
        f"label={rec.get('risk_label')}",
    )

    # 4. Supported choices are constrained by the Streamlit selectboxes
    check(
        "supported crops and windows are constrained",
        set(CROPS) == {"maize", "beans"}
        and PLANTING_WINDOWS == list(range(25, 34)),
    )

    # 5. Full season scan ranks every crop x window option
    scan = service.season_scan()
    all_options = scan["all_options"]
    check(
        "season scan ranks all crop-window combinations",
        scan["recommendation"]["risk_label"] in VALID_LABELS
        and len(all_options) == len(CROPS) * len(PLANTING_WINDOWS),
    )

    # 6. Streamlit module imports and can read the metrics report
    try:
        streamlit_app = importlib.import_module("streamlit_app")
        metrics = streamlit_app.load_metrics()
        bounds = streamlit_app.CLIMATE_INPUT_BOUNDS
        ok = (
            metrics is not None
            and "xgb_full" in metrics
            and bounds["cum_rain_since_sep1"] == (0.0, 800.0)
            and bounds["last_dekad_rain"] == (0.0, 300.0)
            and bounds["last3_rain"] == (0.0, 500.0)
            and bounds["pre_tmax_anom"] == (-3.0, 3.0)
        )
        detail = "imported"
    except ModuleNotFoundError as exc:
        ok = False
        detail = f"missing dependency: {exc.name}"
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    check("streamlit_app imports, loads metrics, and constrains climate inputs",
          ok, detail)


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
