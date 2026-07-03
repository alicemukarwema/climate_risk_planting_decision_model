"""
Scenario and performance checks for the Streamlit capstone app.

Run:
    .venv/bin/python tests/test_scenarios_performance.py

The script exercises the same model service used by the Streamlit app, using
multiple crops, planting windows, what-if climate values, UI-constrained input
choices, and a lightweight latency check.
"""
import os
import platform
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from crops import CROPS, PLANTING_WINDOWS
from model import MODELS
from service import AdvisoryService


VALID_LABELS = {"suitable", "risky", "delay"}


def scenario_result(
    service: AdvisoryService,
    name: str,
    crop: str,
    window: int,
    overrides: dict | None = None,
) -> dict:
    started = time.perf_counter()
    rec = service.predict_option(crop, window, overrides)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "name": name,
        "crop": crop,
        "window": window,
        "label": rec["risk_label"],
        "confidence": rec["confidence"],
        "risk_score": round(rec["risk_score"], 3),
        "latency_ms": round(elapsed_ms, 1),
        "passed": rec["risk_label"] in VALID_LABELS
        and bool(rec["explanation"])
        and bool(rec["class_probabilities"]),
    }


def print_table(rows: list[dict]) -> None:
    print(f"{'SCENARIO':26s} {'CROP':8s} {'WINDOW':9s} "
          f"{'LABEL':9s} {'CONF':>6s} {'RISK':>6s} {'MS':>8s} RESULT")
    print("-" * 92)
    for row in rows:
        print(
            f"{row['name']:26s} {row['crop']:8s} {str(row['window']):9s} "
            f"{row['label']:9s} {row['confidence']:6.3f} "
            f"{row['risk_score']:6.3f} {row['latency_ms']:8.1f} "
            f"{'PASS' if row['passed'] else 'FAIL'}"
        )


def main() -> int:
    service = AdvisoryService()
    checks = []

    checks.append(("model artefact readiness", AdvisoryService.artefacts_present()))

    report_path = MODELS / "report.json"
    metrics = json.loads(report_path.read_text())
    selected = metrics["xgb_full"]
    expected_models = {"rule_baseline", "dt_raw", "dt_risk", "xgb_full"}
    checks.append(("metrics/model comparison", expected_models <= set(metrics)
                   and selected["macro_f1"] is not None
                   and selected["balanced_accuracy"] is not None))

    scenarios = [
        (
            "maize early normal",
            "maize",
            25,
            None,
        ),
        (
            "beans mid-season",
            "beans",
            28,
            None,
        ),
        (
            "dry delayed onset",
            "maize",
            31,
            {
                "cum_rain_since_sep1": 12,
                "last_dekad_rain": 3,
                "last3_rain": 14,
                "onset_reached": False,
            },
        ),
        (
            "wet observed season",
            "beans",
            29,
            {
                "cum_rain_since_sep1": 145,
                "last_dekad_rain": 38,
                "last3_rain": 92,
                "onset_reached": True,
            },
        ),
        (
            "heat-stress what-if",
            "beans",
            30,
            {"pre_tmax_anom": 2.4},
        ),
    ]
    scenario_rows = [scenario_result(service, name, crop, window, overrides)
                     for name, crop, window, overrides in scenarios]
    started = time.perf_counter()
    scan = service.season_scan()
    elapsed_ms = (time.perf_counter() - started) * 1000
    rec = scan["recommendation"]
    scenario_rows.append({
        "name": "all windows scan",
        "crop": "both",
        "window": "25-33",
        "label": rec["risk_label"],
        "confidence": rec["confidence"],
        "risk_score": round(rec["risk_score"], 3),
        "latency_ms": round(elapsed_ms, 1),
        "passed": rec["risk_label"] in VALID_LABELS
        and len(scan["all_options"]) == len(CROPS) * len(PLANTING_WINDOWS),
    })
    checks.append(("different data values", all(r["passed"] for r in scenario_rows)))

    checks.append(("ui-constrained inputs", set(CROPS) == {"maize", "beans"}
                   and PLANTING_WINDOWS == list(range(25, 34))))

    # Latency is measured after startup/model loading because that is what a
    # Streamlit user experiences after the cached service is warm.
    timings = []
    for _ in range(10):
        started = time.perf_counter()
        service.predict_option("maize", 25)
        timings.append((time.perf_counter() - started) * 1000)
    avg_ms = statistics.mean(timings)
    p95_ms = sorted(timings)[int(len(timings) * 0.95) - 1]
    checks.append(("warm prediction performance", p95_ms < 1500))

    print("\nENVIRONMENT")
    print("-" * 92)
    print(f"OS/software: {platform.platform()}")
    print(f"Python:      {platform.python_version()}")
    print(f"CPU cores:   {os.cpu_count()}")
    print("App:         Streamlit demo")
    print("Model:       Climate Risk-Aware Planting Window Classifier")

    print("\nVALIDATION CHECKS")
    print("-" * 92)
    for name, passed in checks:
        print(f"{name:34s} {'PASS' if passed else 'FAIL'}")

    print("\nDIFFERENT INPUT VALUES")
    print("-" * 92)
    print_table(scenario_rows)

    print("\nPERFORMANCE")
    print("-" * 92)
    print(f"Warm prediction average latency: {avg_ms:.1f} ms")
    print(f"Warm prediction p95 latency:     {p95_ms:.1f} ms")

    passed = sum(1 for _, ok in checks if ok)
    print("\nSUMMARY")
    print("-" * 92)
    print(f"{passed}/{len(checks)} validation checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
