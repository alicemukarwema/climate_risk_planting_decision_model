import streamlit as st

st.set_page_config(page_title="Terms & Privacy", page_icon="📄")

st.title("Terms of Use & Privacy Policy")
st.caption("Climate Risk-Aware Planting Recommendation Model — research prototype")

st.header("Terms of Use")
st.markdown("""
- **Purpose and nature:** A research prototype providing climate risk-aware
  planting-window decision support for maize and beans in Nyagatare District,
  hosted publicly on Streamlit for demonstration and review. It is not a finished
  product in operational use.
- **No guarantee of outcome:** Recommendations, risk scores and labels are
  probabilistic estimates. They do **not** guarantee rainfall, crop establishment
  or yield. The final planting decision remains with the user.
- **Acceptable use:** For evaluation and educational review only — not commercial
  advisory resale, and not a substitute for professional agronomic advice.
- **Limitation of liability:** The developer is not liable for losses arising from
  decisions made using these outputs.
- **Changes:** The terms and the underlying model may change during the research period.
""")

st.header("Privacy Policy")
st.markdown("""
- **What is collected:** The app takes planting-scenario inputs (crop, sector or
  location, planting window and climate indicators) to produce a prediction. It does
  **not** request or store any name, phone number or other identifying information.
- **Prediction inputs:** Used only to generate the on-screen result and not retained
  after the session ends.
- **Hosting:** The app runs on Streamlit Community Cloud, so only standard
  platform-level hosting logs apply, handled by Streamlit rather than the researcher.
- **Reviewer feedback:** During the planned evaluation, feedback is collected
  separately and only with voluntary, informed consent, and is kept as anonymous
  ratings of output clarity — no names, phone numbers or personal comments are stored.
- **Legal basis, rights and contact:** Processing aligns with Rwanda Law No. 058/2021.
  Participants may withdraw and request deletion of their feedback at any time.
  Contact: **[your ALU email]**.
""")

st.info("This tool is decision support, not a guarantee of rainfall or harvest. "
        "Final decisions rest with the user.")
