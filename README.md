# Polynomial Regression for Clinical Biomarker Modeling

This repository contains the implementation of **Polynomial Regression** for modeling non-linear pharmacodynamic relationships (Drug Dosage vs. Patient Recovery Index). 

Developed for the **Machine Learning (ECS-522)** curriculum, this model aligns as the quantitative biomarker trend estimation component for an **Intelligent Medical Assistant using Large Language Models (RAG)**.

---

## 📌 Problem Formulation
In clinical diagnostics, biological responses rarely follow a linear trajectory. While standard Linear Regression ($y = \theta_0 + \theta_1 x$) assumes a constant rate of change, actual pharmacodynamics feature an optimal therapeutic window followed by a decline due to toxicity or receptor saturation.

Polynomial Regression projects the input feature $x$ (Drug Dosage in mg) into higher-degree polynomial space:

$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d$$

The parameters $\boldsymbol{\theta}$ are calculated analytically via the closed-form Normal Equation:

$$\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

---

## 🔬 Model Comparison & Bias-Variance Analysis

| Model Complexity | Mathematical Degree | Machine Learning Behavior | Clinical Interpretation |
| :--- | :---: | :--- | :--- |
| **Linear Model** | Degree 1 | **Underfitting (High Bias)** | Fails to detect the peak therapeutic recovery window. |
| **Quadratic Model** | Degree 2 | **Optimal Fit (Balanced)** | Accurately models the natural peak recovery (around 5.5–6.0 mg). |
| **High-Order Model** | Degree 9 | **Overfitting (High Variance)** | Memorizes observation noise; oscillates wildly near boundaries. |

---

## 🏥 Alignment with M.Tech Dissertation
* **Dissertation Topic:** *RAG Based Intelligent Medical Assistant Using Large Language Models*
* **Role in System:**
  * **Qualitative Reasoning (RAG + LLM):** Retrieves peer-reviewed clinical guidelines, contraindications, and protocol steps from indexed medical documents.
  * **Quantitative Modeling (Polynomial Regression):** Analyzes continuous patient vitals and plots optimal drug dosage curves.
  * **Combined Output:** Provides an evidence-grounded prescription dosage backed by both clinical literature and mathematical regression trends.

---

## 🚀 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/polynomial-regression-clinical-modeling.git](https://github.com/YOUR_USERNAME/polynomial-regression-clinical-modeling.git)
   cd polynomial-regression-clinical-modeling
