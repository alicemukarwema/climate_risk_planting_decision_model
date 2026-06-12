"""
Deployment tests (capstone checklist section 10).

Run:  python tests/test_api.py        (no server needed - uses TestClient)

Covers: valid inputs, missing inputs, invalid inputs, forced maize, forced
beans, a risky output, a delay output, the season scan, and the metrics
report. Results are printed as a PASS/FAIL table.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from app import app

REQUIRED_FIELDS = {"crop", "planting_window", "risk_label",
                   "class_probabilities", "confidence", "risk_score",
                   "p_rain_sufficient", "p_dry_spell", "p_temp_stress",
                   "explanation"}
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))


with TestClient(app) as client:          # context manager triggers startup
    # 1. health
    r = client.get("/health")
    check("GET /health returns ok", r.status_code == 200
          and r.json()["artefacts_present"])

    # 2. valid full scenario
    r = client.post("/predict", json={"crop": "auto", "last3_rain": 60,
                                      "pre_tmax_anom": 0.0})
    rec = r.json()["recommendation"]
    check("valid input -> 200 + all required output fields",
          r.status_code == 200 and REQUIRED_FIELDS <= set(rec),
          f"got {rec['crop']} / {rec['planting_window']} / {rec['risk_label']}")

    # 3. missing inputs -> climatological defaults
    r = client.post("/predict", json={})
    check("missing inputs handled (falls back to climatology)",
          r.status_code == 200
          and r.json()["recommendation"]["risk_label"] in
          ("suitable", "risky", "delay"))

    # 4. invalid input rejected
    r = client.post("/predict", json={"crop": "auto",
                                      "window_start_dekad": 50})
    check("invalid window (50) rejected with 422", r.status_code == 422)

    # 5. maize recommendation
    r = client.post("/predict", json={"crop": "maize",
                                      "window_start_dekad": 25})
    rec = r.json()["recommendation"]
    check("forced maize assessment returns maize", rec["crop"] == "maize",
          f"label={rec['risk_label']} score={rec['risk_score']}")

    # 6. beans recommendation
    r = client.post("/predict", json={"crop": "beans",
                                      "window_start_dekad": 28})
    rec = r.json()["recommendation"]
    check("forced beans assessment returns beans", rec["crop"] == "beans",
          f"label={rec['risk_label']} score={rec['risk_score']}")

    # 7. risky output reachable (dry recent spell + hot year, early window)
    r = client.post("/predict", json={"crop": "maize",
                                      "window_start_dekad": 25,
                                      "last3_rain": 10, "last_dekad_rain": 2,
                                      "onset_reached": False,
                                      "pre_tmax_anom": 1.5})
    lbl = r.json()["recommendation"]["risk_label"]
    check("stress scenario produces non-suitable label",
          lbl in ("risky", "delay"), f"label={lbl}")

    # 8. delay output reachable (late-November beans)
    r = client.post("/predict", json={"crop": "beans",
                                      "window_start_dekad": 33})
    lbl = r.json()["recommendation"]["risk_label"]
    check("late-November beans -> delay", lbl == "delay", f"label={lbl}")

    # 9. season scan: 2 crops x 9 windows
    r = client.get("/recommend/season")
    check("season scan returns 18 ranked options",
          r.status_code == 200 and len(r.json()["all_options"]) == 18)

    # 10. metrics report: all four proposal models
    r = client.get("/metrics")
    check("metrics report contains all 4 models",
          {"rule_baseline", "dt_raw", "dt_risk", "xgb_full"}
          <= set(r.json()))

    # 11. prediction log written
    check("prediction_logs.csv written",
          (ROOT / "data" / "prediction_logs.csv").exists())

print(f"\n{'TEST':58s} RESULT")
print("-" * 72)
ok = True
for name, passed, detail in results:
    ok &= passed
    print(f"{name:58s} {'PASS' if passed else 'FAIL'}"
          + (f"   [{detail}]" if detail else ""))
print("-" * 72)
print(f"{sum(p for _, p, _ in results)}/{len(results)} passed")
sys.exit(0 if ok else 1)
