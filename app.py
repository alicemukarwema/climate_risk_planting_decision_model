"""
Serving layer (proposal objective 6): FastAPI app exposing the climate
risk-aware planting recommendation model.

Run:    uvicorn app:app --reload --port 8000
API:    http://localhost:8000/            (redirects to Swagger UI)
UI:     http://localhost:8000/web         (optional simple advisory web page)

Endpoints
  GET  /health               liveness + artefact check
  GET  /recommend/season     rank all crop x window options for Season A
  POST /predict              risk-aware assessment of one scenario
  GET  /metrics              4-model comparison report (proposal Table 6)
"""
import sys, json
from pathlib import Path
from typing import Literal, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crops import PLANTING_WINDOWS, DEKAD_LABEL
from model import MODELS
from service import AdvisoryService

DATA_SOURCE = "Meteo Rwanda / ENACTS Maproom dekadal extracts"
DECISION_SUPPORT_WARNING = "Decision support only, not guaranteed farming advice."
SELECTED_MODEL_KEY = "xgb_full"
SELECTED_MODEL_NAME = "Climate Risk-Aware Planting Window Classifier"
MISSING_ARTEFACTS_MESSAGE = (
    "Model artifacts missing. Run `python train.py` first."
)

app = FastAPI(
    title="Nyagatare Climate Risk-Aware Planting Advisor",
    description="Stochastic simulation (Markov chain + Monte Carlo) + "
                "XGBoost classification of Season A planting windows "
                "(suitable / risky / delay) for maize and beans, "
                "Nyagatare District, Rwanda. Decision support only.",
    version="1.0.0")

service: AdvisoryService | None = None


@app.on_event("startup")
def _startup():
    global service
    if AdvisoryService.artefacts_present():
        service = AdvisoryService()


def _svc() -> AdvisoryService:
    if service is None:
        raise HTTPException(503, MISSING_ARTEFACTS_MESSAGE)
    return service


def _report_path() -> Path:
    return MODELS / "report.json"


def _selected_model(report: dict | None = None) -> str | None:
    model_file = MODELS / "xgb_planting_risk.json"
    if report is not None and SELECTED_MODEL_KEY in report:
        return SELECTED_MODEL_NAME
    if model_file.exists():
        return SELECTED_MODEL_NAME
    return None


def _load_report() -> dict:
    report = _report_path()
    if not report.exists():
        raise HTTPException(503, MISSING_ARTEFACTS_MESSAGE)
    return json.loads(report.read_text())


def _model_comparison_table(report: dict) -> list[dict]:
    rows = []
    for model_name, values in report.items():
        if model_name.startswith("_") or not isinstance(values, dict):
            continue
        rows.append({
            "model": model_name,
            "macro_f1": values.get("macro_f1"),
            "balanced_accuracy": values.get("balanced_accuracy"),
            "brier_score": values.get("brier_score"),
        })
    return rows


class Scenario(BaseModel):
    """A what-if planting scenario. Every climate field is optional -
    anything you omit falls back to the Nyagatare climatology."""
    crop: Literal["maize", "beans", "auto"] = Field(
        "auto", description="Crop to assess; 'auto' compares both")
    window_start_dekad: Optional[int] = Field(
        None, ge=25, le=33,
        description="Candidate planting dekad (25 = 1-10 Sep ... "
                    "33 = 21-30 Nov); omit to scan all windows")
    cum_rain_since_sep1: Optional[float] = Field(
        None, ge=0, le=800, description="Rain observed since 1 Sep (mm)")
    last_dekad_rain: Optional[float] = Field(
        None, ge=0, le=300, description="Rain in the last dekad (mm)")
    last3_rain: Optional[float] = Field(
        None, ge=0, le=500, description="Rain over the last 3 dekads (mm)")
    onset_reached: Optional[bool] = Field(
        None, description="Has a >=25 mm dekad occurred since 1 Sep?")
    pre_tmax_anom: Optional[float] = Field(
        None, ge=-3, le=3,
        description="May-Aug max-temperature anomaly vs normal (deg C); "
                    "0 = normal year")


@app.get("/health")
def health():
    artefacts_present = AdvisoryService.artefacts_present()
    model_loaded = service is not None
    report = None
    if _report_path().exists():
        report = json.loads(_report_path().read_text())
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "selected_model": _selected_model(report),
        "data_source": DATA_SOURCE,
        "warning": DECISION_SUPPORT_WARNING,
        # Kept for the existing deployment test and backwards compatibility.
        "artefacts_present": artefacts_present,
    }


@app.get("/recommend/season")
def recommend_season():
    """Rank every crop x planting-window option for Season A under
    climatological conditions and return the best risk-aware choice."""
    result = _svc().season_scan()
    _svc().log_prediction(result["recommendation"])
    return result


@app.post("/predict")
def predict(s: Scenario):
    """Risk-aware assessment of a planting scenario. Returns the
    recommended crop, planting window, risk label, class probabilities,
    confidence, stochastic risk components, and a plain-language
    explanation (proposal objective 6)."""
    svc = _svc()
    overrides = s.model_dump(exclude={"crop", "window_start_dekad"})
    if overrides.get("onset_reached") is not None:
        overrides["onset_reached"] = int(overrides["onset_reached"])
    crops = ["maize", "beans"] if s.crop == "auto" else [s.crop]
    windows = PLANTING_WINDOWS if s.window_start_dekad is None \
        else [s.window_start_dekad]
    options = [svc.predict_option(c, w, overrides)
               for c in crops for w in windows]
    from recommend import pick_best
    result = pick_best(options)
    result["scenario"] = s.model_dump()
    svc.log_prediction(result["recommendation"])
    return result


@app.get("/metrics")
def metrics():
    """Model comparison report: rule baseline vs Decision Trees vs XGBoost
    (macro F1, balanced accuracy, Brier score, per-class precision/recall,
    confusion matrices, feature importance)."""
    report = _load_report()
    response = dict(report)
    selected = report.get(SELECTED_MODEL_KEY, {})
    response.update({
        "selected_model": _selected_model(report),
        "model_comparison_table": _model_comparison_table(report),
        "selected_model_metrics": {
            "macro_f1": selected.get("macro_f1"),
            "balanced_accuracy": selected.get("balanced_accuracy"),
            "brier_score": selected.get("brier_score"),
            "per_class": selected.get("per_class"),
            "confusion_matrix": selected.get("confusion_matrix"),
            "confusion_matrix_labels": selected.get("confusion_matrix_labels"),
        },
        "note": (
            "Delay/high-risk recall matters because false reassurance is the "
            "most serious error: predicting suitable when conditions are "
            "actually delay or high risk."
        ),
    })
    return response


@app.get("/windows")
def windows():
    """Candidate Season A planting windows (dekad id -> calendar label)."""
    return {w: DEKAD_LABEL[w] for w in PLANTING_WINDOWS}


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/web", include_in_schema=False)
def web_ui():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
