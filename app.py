import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from groq import Groq

# Page Config
st.set_page_config(page_title="Clinical Dosage & AI Assistant", page_icon="💊", layout="wide")
st.title("💊 Clinical Pharmacodynamics & AI Medical Assistant")
st.markdown("Integration of **Quantitative Layer (Polynomial Regression)** & **Qualitative Layer (LLM Clinical Guidance)**")

# --- SIDEBAR: 1. DATASET SELECTION ---
st.sidebar.header("📂 1. Dataset Selection")
uploaded_file = st.sidebar.file_uploader("Upload Custom Patient CSV (Optional)", type=["csv"])
st.sidebar.caption("CSV must have two columns: `dosage` and `recovery`")

@st.cache_data
def get_default_data():
    np.random.seed(42)
    n_samples = 35
    dosage = np.sort(np.random.uniform(1.0, 10.0, n_samples)).reshape(-1, 1)
    recovery = -1.2 * (dosage ** 2) + 14.0 * dosage - 10.0 + np.random.normal(0, 3.2, size=dosage.shape)
    return dosage, recovery

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        dosage = df[['dosage']].values
        recovery = df[['recovery']].values
        st.sidebar.success(f"Loaded {len(df)} custom patient records!")
    except Exception:
        st.sidebar.error("Error reading CSV. Ensure columns are named 'dosage' and 'recovery'.")
        dosage, recovery = get_default_data()
else:
    dosage, recovery = get_default_data()

# Train Models Dynamically
poly2 = PolynomialFeatures(degree=2, include_bias=False)
model_deg1 = LinearRegression().fit(dosage, recovery)
model_deg2 = LinearRegression().fit(poly2.fit_transform(dosage), recovery)
model_deg9 = make_pipeline(PolynomialFeatures(degree=9, include_bias=False), StandardScaler(), LinearRegression())
model_deg9.fit(dosage, recovery)

# Calculate Optimal Peak Dose (-theta_1 / 2*theta_2)
theta_0 = model_deg2.intercept_[0]
theta_1 = model_deg2.coef_[0][0]
theta_2 = model_deg2.coef_[0][1]
optimal_dose = -theta_1 / (2 * theta_2) if theta_2 < 0 else float(np.max(dosage))

# --- SIDEBAR: 2. SIMULATION CONTROLS ---
st.sidebar.header("🎛️ 2. Dosage Simulation")
min_d, max_d = float(np.min(dosage)), float(np.max(dosage))
default_val = float(np.clip(optimal_dose, min_d, max_d))
selected_dose = st.sidebar.slider("Select Drug Dosage (mg):", min_value=min_d, max_value=max_d, value=default_val, step=0.05)

st.sidebar.subheader("Model Curves:")
show_deg1 = st.sidebar.checkbox("Degree 1 (Underfit)", value=False)
show_deg2 = st.sidebar.checkbox("Degree 2 (Optimal Fit)", value=True)
show_deg9 = st.sidebar.checkbox("Degree 9 (Overfit)", value=False)

# --- SIDEBAR: 3. GROQ AI INTEGRATION ---
st.sidebar.header("🤖 3. AI Clinical Layer")
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("✅ AI Assistant Ready (Key Loaded)")
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password", help="Get a free key from console.groq.com")

# --- MAIN DASHBOARD: QUANTITATIVE LAYER ---
pred_recovery = model_deg2.predict(poly2.transform([[selected_dose]]))[0][0]

col1, col2, col3 = st.columns(3)
col1.metric("Selected Dosage", f"{selected_dose:.2f} mg")
col2.metric("Predicted Recovery Index", f"{pred_recovery:.2f}")
col3.metric("Calculated Optimal Peak", f"{optimal_dose:.2f} mg")

# Clinical Alert Logic
if selected_dose < 3.0:
    zone_status = "Sub-Therapeutic Zone (< 3 mg)"
    st.warning(f"⚠️ **{zone_status}:** Minimal clinical recovery expected. Dosage is too low.")
elif 3.0 <= selected_dose <= 8.0:
    zone_status = "Optimal Therapeutic Window (3 - 8 mg)"
    st.success(f"✅ **{zone_status}:** Safe and effective recovery zone.")
else:
    zone_status = "Toxicity Zone (> 8 mg)"
    st.error(f"🚨 **{zone_status}:** Sharp efficacy drop and high risk of drug toxicity!")

# Layout: Plot on Left, AI Assistant on Right
left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.subheader("📈 Pharmacodynamic Curve")
    X_grid = np.linspace(min_d, max_d, 300).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(dosage, recovery, color='#1e293b', s=45, label='Patient Trials', zorder=3)

    if show_deg1:
        ax.plot(X_grid, model_deg1.predict(X_grid), color='#dc2626', linestyle='--', label='Degree 1: Underfit')
    if show_deg2:
        ax.plot(X_grid, model_deg2.predict(poly2.transform(X_grid)), color='#16a34a', linewidth=2.5, label='Degree 2: Optimal')
    if show_deg9:
        ax.plot(X_grid, model_deg9.predict(X_grid), color='#2563eb', linestyle=':', label='Degree 9: Overfit')

    ax.scatter([selected_dose], [pred_recovery], color='#f59e0b', s=150, edgecolor='black', zorder=5, label='Selected Dose')
    ax.axvline(x=selected_dose, color='#f59e0b', linestyle='-.', alpha=0.6)
    ax.set_xlabel("Drug Dosage (mg)")
    ax.set_ylabel("Recovery / Efficacy Index")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    st.pyplot(fig)

with right_col:
    st.subheader("🩺 Qualitative AI Clinical Guidance")
    st.info(f"**Learned Quadratic Equation:**\n`y = {theta_0:.2f} + ({theta_1:.2f})x + ({theta_2:.2f})x²`")

    if st.button("Generate Clinical Assessment", use_container_width=True):
        if not groq_api_key:
            st.warning("Please enter your Groq API Key in the left sidebar to activate the AI Assistant.")
        else:
            try:
                client = Groq(api_key=groq_api_key)
                prompt = f"""
                You are an AI Clinical Decision Support Assistant.
                A polynomial regression pharmacodynamic model analyzed a patient drug dosage:
                - Selected Dosage: {selected_dose:.2f} mg
                - Optimal Peak Dosage: {optimal_dose:.2f} mg
                - Predicted Recovery Index: {pred_recovery:.2f}
                - Clinical Status: {zone_status}

                Provide a concise, professional clinical interpretation (3-4 bullet points) covering:
                1. Efficacy expectation at this dosage.
                2. Safety bounds / toxicity risk or under-dosing risk.
                3. Recommended clinical monitoring or adjustment.
                """
                with st.spinner("Analyzing clinical parameters..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                    )
                    st.markdown(chat_completion.choices[0].message.content)
            except Exception as e:
                st.error(f"API Error: {e}")
