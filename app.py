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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal, Optional

from crops import PLANTING_WINDOWS, DEKAD_LABEL
from model import MODELS
from service import AdvisoryService

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
        raise HTTPException(503, "Model artefacts missing - run "
                                 "`python train.py` first.")
    return service


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
    return {"status": "ok",
            "artefacts_present": AdvisoryService.artefacts_present()}


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
    report = MODELS / "report.json"
    if not report.exists():
        raise HTTPException(503, "Run `python train.py` first.")
    return json.loads(report.read_text())


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
