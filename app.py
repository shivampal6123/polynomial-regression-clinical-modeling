import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# Page Config
st.set_page_config(page_title="Clinical Dosage Simulator", page_icon="💊", layout="wide")
st.title("💊 Clinical Pharmacodynamics: Drug Dosage & Recovery Simulator")
st.markdown("Non-Linear Polynomial Regression Model for Optimal Therapeutic Window Prediction")

# 1. Data Generation & Model Training
@st.cache_resource
def train_models():
    np.random.seed(42)
    n_samples = 35
    dosage = np.sort(np.random.uniform(1.0, 10.0, n_samples)).reshape(-1, 1)
    recovery = -1.2 * (dosage ** 2) + 14.0 * dosage - 10.0 + np.random.normal(0, 3.2, size=dosage.shape)
    
    # Degree 1
    m1 = LinearRegression().fit(dosage, recovery)
    
    # Degree 2 (Optimal)
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    m2 = LinearRegression().fit(poly2.fit_transform(dosage), recovery)
    
    # Degree 9 (Overfit)
    m9 = make_pipeline(PolynomialFeatures(degree=9, include_bias=False), StandardScaler(), LinearRegression())
    m9.fit(dosage, recovery)
    
    return dosage, recovery, m1, poly2, m2, m9

dosage, recovery, model_deg1, poly2, model_deg2, model_deg9 = train_models()

# Calculate Optimal Peak Dose (-theta_1 / 2*theta_2)
theta_1 = model_deg2.coef_[0][0]
theta_2 = model_deg2.coef_[0][1]
optimal_dose = -theta_1 / (2 * theta_2)

# 2. Sidebar Controls
st.sidebar.header("🎛️ Simulation Controls")
selected_dose = st.sidebar.slider("Select Drug Dosage (mg):", min_value=1.0, max_value=10.0, value=5.84, step=0.05)

st.sidebar.subheader("Show Model Curves:")
show_deg1 = st.sidebar.checkbox("Degree 1 (Underfit)", value=False)
show_deg2 = st.sidebar.checkbox("Degree 2 (Optimal Fit)", value=True)
show_deg9 = st.sidebar.checkbox("Degree 9 (Overfit)", value=False)

# 3. Prediction & Clinical Zone Logic
pred_recovery = model_deg2.predict(poly2.transform([[selected_dose]]))[0][0]

col1, col2, col3 = st.columns(3)
col1.metric("Selected Dosage", f"{selected_dose:.2f} mg")
col2.metric("Predicted Recovery Index", f"{pred_recovery:.2f}")
col3.metric("Calculated Optimal Peak", f"{optimal_dose:.2f} mg")

# Clinical Alert Box
if selected_dose < 3.0:
    st.warning("⚠️ **Sub-Therapeutic Zone (< 3 mg):** Minimal clinical recovery expected. Dosage is too low.")
elif 3.0 <= selected_dose <= 8.0:
    st.success("✅ **Optimal Therapeutic Window (3 - 8 mg):** Safe and effective recovery zone.")
else:
    st.error("🚨 **Toxicity Zone (> 8 mg):** Sharp efficacy drop and high risk of drug toxicity!")

# 4. Interactive Plot
X_grid = np.linspace(1.0, 10.0, 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.scatter(dosage, recovery, color='#1e293b', s=45, label='Patient Trials (n=35)', zorder=3)

if show_deg1:
    ax.plot(X_grid, model_deg1.predict(X_grid), color='#dc2626', linestyle='--', label='Degree 1: Underfit')
if show_deg2:
    ax.plot(X_grid, model_deg2.predict(poly2.transform(X_grid)), color='#16a34a', linewidth=2.5, label='Degree 2: Optimal')
if show_deg9:
    ax.plot(X_grid, model_deg9.predict(X_grid), color='#2563eb', linestyle=':', label='Degree 9: Overfit')

# Highlight User's Selected Point
ax.scatter([selected_dose], [pred_recovery], color='#f59e0b', s=150, edgecolor='black', zorder=5, label='Current Simulation')
ax.axvline(x=selected_dose, color='#f59e0b', linestyle='-.', alpha=0.6)

ax.set_title("Clinical Pharmacodynamics Curve", fontweight='bold')
ax.set_xlabel("Drug Dosage (mg)")
ax.set_ylabel("Recovery / Efficacy Index")
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()

st.pyplot(fig)
