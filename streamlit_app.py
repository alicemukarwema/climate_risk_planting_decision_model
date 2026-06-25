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


st.set_page_config(
    page_title="Climate Risk-Aware Planting Window Classifier",
    layout="centered",
)


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
        "This demo classifies maize and bean planting windows in Nyagatare "
        "District as suitable, risky, or delay using Meteo Rwanda/ENACTS "
        "dekadal climate data and a trained machine-learning model."
    )

    artefacts_present = AdvisoryService.artefacts_present()
    if not artefacts_present:
        st.error("Model artifacts are missing. Run `python train.py` first.")

    if artefacts_present:
        with st.form("prediction_form"):
            crop = st.selectbox("Crop", ["maize", "beans"])
            window = st.selectbox(
                "Season A planting window",
                PLANTING_WINDOWS,
                format_func=lambda value: DEKAD_LABEL[value],
            )
            st.text_input("Location", value="Nyagatare District", disabled=True)
            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            with st.spinner("Classifying planting-window risk..."):
                service = load_service()
                result = service.predict_option(crop, window)
            show_prediction(result)

    show_metrics(load_metrics())

    st.subheader("Limitations")
    st.markdown(
        "- The model uses Meteo Rwanda/ENACTS dekadal climate extracts.\n"
        "- The data are spatial averages, not farm-level measurements.\n"
        "- The labels are proxy agronomic labels, not measured yield labels.\n"
        "- The output is decision support only, not guaranteed farming advice.\n"
        "- Mobile app, SMS/USSD, dashboard, and automatic retraining are future work."
    )


if __name__ == "__main__":
    main()
