"""Streamlit demo for the planting-window risk classifier.

Run:
    streamlit run streamlit_app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from crops import DEKAD_LABEL, PLANTING_WINDOWS
from model import MODELS
from service import AdvisoryService


SELECTED_MODEL_KEY = "xgb_full"
SELECTED_MODEL_NAME = "Climate Risk-Aware Planting Window Classifier"
CLIMATE_INPUT_BOUNDS = {
    "cum_rain_since_sep1": (0.0, 800.0),
    "last_dekad_rain": (0.0, 300.0),
    "last3_rain": (0.0, 500.0),
    "pre_tmax_anom": (-3.0, 3.0),
}

st.set_page_config(
    page_title="Climate Risk-Aware Planting Window Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.page_link("pages/2_Terms_and_Privacy.py", label="📄 Terms of Use & Privacy Policy")

@st.cache_resource(show_spinner=False)
def load_service() -> AdvisoryService:
    return AdvisoryService()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    report_path = MODELS / "report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text())


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def show_metrics(report: dict | None) -> None:
    st.subheader("Model Metrics")
    if report is None:
        st.info("Model metrics are not available. Run `python train.py` first.")
        return

    selected = report.get(SELECTED_MODEL_KEY, {})
    st.write(f"Selected model: **{SELECTED_MODEL_NAME}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Macro F1", selected.get("macro_f1", "N/A"))
    c2.metric("Balanced accuracy", selected.get("balanced_accuracy", "N/A"))
    c3.metric("Brier score", selected.get("brier_score", "N/A"))

    per_class = selected.get("per_class", {})
    if per_class:
        st.write("Per-class recall")
        recall_rows = [
            {"class": label, "recall": values.get("recall")}
            for label, values in per_class.items()
        ]
        st.dataframe(recall_rows, hide_index=True, use_container_width=True)


def show_prediction(result: dict) -> None:
    st.subheader("Prediction")

    c1, c2, c3 = st.columns(3)
    c1.metric("Crop", result["crop"])
    c2.metric("Planting window", result["planting_window"])
    c3.metric("Risk class", result["risk_label"])

    st.write("Class probabilities")
    probability_rows = [
        {"class": label, "probability": prob}
        for label, prob in result["class_probabilities"].items()
    ]
    st.dataframe(probability_rows, hide_index=True, use_container_width=True)

    st.write("Risk components")
    r1, r2, r3 = st.columns(3)
    r1.metric("Rainfall sufficient", format_percent(result.get("p_rain_sufficient")))
    r2.metric("Dry-spell risk", format_percent(result.get("p_dry_spell")))
    r3.metric("Temperature stress", format_percent(result.get("p_temp_stress")))

    st.write("Explanation")
    st.info(result["explanation"])


def main() -> None:
    st.title("Climate Risk-Aware Planting Window Classifier")
    st.write(
        "This classifies maize and bean planting windows in Nyagatare "
        "District as suitable, risky, or delay using Meteo Rwanda/ENACTS "
        "dekadal climate data and a trained machine-learning model."
    )

    artefacts_present = AdvisoryService.artefacts_present()
    if not artefacts_present:
        st.error("Model artifacts are missing. Run `python train.py` first.")

    if artefacts_present:
        service = load_service()
        with st.form("prediction_form"):
            crop = st.selectbox("Crop", ["maize", "beans"])
            window = st.selectbox(
                "Season A planting window",
                PLANTING_WINDOWS,
                format_func=lambda value: DEKAD_LABEL[value],
            )
            st.text_input("Location", value="Nyagatare District", disabled=True)
            use_observed = st.checkbox("Use observed climate values", value=False)

            overrides = None
            if use_observed:
                defaults = service.defaults[window]
                cum_rain = st.number_input(
                    "Rain observed since 1 Sep (mm)",
                    min_value=CLIMATE_INPUT_BOUNDS["cum_rain_since_sep1"][0],
                    max_value=CLIMATE_INPUT_BOUNDS["cum_rain_since_sep1"][1],
                    value=float(defaults["cum_rain_since_sep1"]),
                    step=1.0,
                )
                last_dekad_rain = st.number_input(
                    "Rain in last dekad (mm)",
                    min_value=CLIMATE_INPUT_BOUNDS["last_dekad_rain"][0],
                    max_value=CLIMATE_INPUT_BOUNDS["last_dekad_rain"][1],
                    value=float(defaults["last_dekad_rain"]),
                    step=1.0,
                )
                last3_rain = st.number_input(
                    "Rain in last 3 dekads (mm)",
                    min_value=CLIMATE_INPUT_BOUNDS["last3_rain"][0],
                    max_value=CLIMATE_INPUT_BOUNDS["last3_rain"][1],
                    value=float(defaults["last3_rain"]),
                    step=1.0,
                )
                onset_reached = st.checkbox(
                    "Rainfall onset reached",
                    value=bool(defaults["cum_rain_since_sep1"] >= 25),
                )
                pre_tmax_anom = st.slider(
                    "May-Aug max-temperature anomaly (deg C)",
                    min_value=CLIMATE_INPUT_BOUNDS["pre_tmax_anom"][0],
                    max_value=CLIMATE_INPUT_BOUNDS["pre_tmax_anom"][1],
                    value=0.0,
                    step=0.1,
                )
                overrides = {
                    "cum_rain_since_sep1": cum_rain,
                    "last_dekad_rain": last_dekad_rain,
                    "last3_rain": last3_rain,
                    "onset_reached": int(onset_reached),
                    "pre_tmax_anom": pre_tmax_anom,
                }

            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            with st.spinner("Classifying planting-window risk..."):
                result = service.predict_option(crop, window, overrides)
            show_prediction(result)

    show_metrics(load_metrics())


if __name__ == "__main__":
    main()
