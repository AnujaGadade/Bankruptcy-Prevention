import streamlit as st
import pickle
import numpy as np

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Prevention System",
    page_icon="🏦",
    layout="wide"
)

# ----------------------------------------------------
# Load Model and Scaler
# ----------------------------------------------------
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("🏦 Bankruptcy Prevention")

st.sidebar.markdown("---")

st.sidebar.header("📌 About Project")

st.sidebar.write("""
This application predicts whether a company is likely
to face **Bankruptcy** using a Machine Learning model.

The prediction is based on six important business risk
factors that influence financial stability.
""")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Input Features")

st.sidebar.markdown("""
- Industrial Risk
- Management Risk
- Financial Flexibility
- Credibility
- Competitiveness
- Operating Risk
""")

st.sidebar.markdown("---")

st.sidebar.subheader(" Model")

st.sidebar.success("Logistic Regression")

st.sidebar.markdown("---")

st.sidebar.subheader("Developed Using")

st.sidebar.markdown("""
- Python
- Scikit-Learn
- Streamlit
- NumPy
""")

# ----------------------------------------------------
# Main Page
# ----------------------------------------------------
st.title("🏦 Bankruptcy Prevention System")

st.write("""
This web application predicts whether a company is at **Bankruptcy Risk**
or **Non-Bankruptcy** using a trained **Logistic Regression** model.
""")

st.markdown("---")

st.subheader("Enter Company Details")

col1, col2 = st.columns(2)

with col1:

    industrial_risk = st.selectbox(
        "Industrial Risk",
        [0.0, 0.5, 1.0],
        help="0 = Low, 0.5 = Medium, 1 = High"
    )

    management_risk = st.selectbox(
        "Management Risk",
        [0.0, 0.5, 1.0],
        help="0 = Low, 0.5 = Medium, 1 = High"
    )

    financial_flexibility = st.selectbox(
        "Financial Flexibility",
        [0.0, 0.5, 1.0],
        help="0 = Low, 0.5 = Medium, 1 = High"
    )

with col2:

    credibility = st.selectbox(
        "Credibility",
        [0.0, 0.5, 1.0],
        help="0 = Low, 0.5 = Medium, 1 = High"
    )

    competitiveness = st.selectbox(
        "Competitiveness",
        [0.0, 0.5, 1.0],
        help="0 = Low, 0.5 = Medium, 1 = High"
    )

    operating_risk = st.selectbox(
        "Operating Risk",
        [0.0, 0.5, 1.0],
        help="0 = Low, 0.5 = Medium, 1 = High"
    )

st.write("")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
if st.button("Predict Bankruptcy", use_container_width=True):

    input_data = np.array([[
        industrial_risk,
        management_risk,
        financial_flexibility,
        credibility,
        competitiveness,
        operating_risk
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability of both classes
    probability = model.predict_proba(input_scaled)

    bankruptcy_prob = probability[0][1] * 100
    non_bankruptcy_prob = probability[0][0] * 100

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Bankruptcy Risk")

        st.info("""
The model predicts that the company has a **high probability
of financial distress**. Consider reviewing operational,
financial, and management strategies.
""")
    else:
        st.success("✅ Non-Bankruptcy")

        st.info("""
The model predicts that the company appears to be
**financially stable** based on the provided inputs.
""")

    # -------------------------------
    # Prediction Probability
    # -------------------------------
    st.subheader("Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="⚠️ Bankruptcy",
            value=f"{bankruptcy_prob:.2f}%"
        )
        st.progress(bankruptcy_prob / 100)

    with col2:
        st.metric(
            label="✅ Non-Bankruptcy",
            value=f"{non_bankruptcy_prob:.2f}%"
        )
        st.progress(non_bankruptcy_prob / 100)
# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")

st.caption(
    "© 2026 Bankruptcy Prevention System | Logistic Regression | Streamlit"
)