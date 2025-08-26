import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load trained model
model = joblib.load("kidocare_rf_model.pkl")

# Title
st.title("KIDOCARE:Kidney Dyfunction Predictor")

# Animated Marquee Banner
marquee_html = """
<style>
.marquee-container {
    background-color: #f0f8ff;
    padding: 8px;
    border-radius: 5px;
    overflow: hidden;
    white-space: nowrap;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 10px;
}
.marquee-text {
    display: inline-block;
    padding-left: 100%;
    animation: marquee 12s linear infinite;
    font-size: 16px;
    font-weight: bold;
    color: #2c3e50;
}
@keyframes marquee {
    0%   { transform: translate(0, 0); }
    100% { transform: translate(-100%, 0); }
}
</style>
<div class="marquee-container">
    <span class="marquee-text">
        Early detection saves lives. Empowering clinicians to detect kidney damage in children with SCD before it's too late
    </span>
</div>
"""
st.markdown(marquee_html, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["Single Patient", "Batch View"])

# SINGLE PATIENT TAB 
with tab1:
    st.sidebar.header("Enter Patient Data")
    age = st.sidebar.number_input("Age (years)", 0, 18, 10)  # Pediatric range
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    height = st.sidebar.number_input("Height (cm)", 50, 200, 140)
    weight = st.sidebar.number_input("Weight (kg)", 5, 100, 35)
    egfr = st.sidebar.number_input("eGFR (mL/min/1.73m²)", 0.0, 120.0, 80.0)

    # Prepare input to match training features exactly
    input_df = pd.DataFrame({
        "AGE": [age],
        "HEIGHT": [height],
        "WEIGHT": [weight],
        "EGFR": [egfr],
        "SEX_0": [1 if sex == "Male" else 0],
        "SEX_1": [1 if sex == "Female" else 0]
    })

    # Predict
    prediction = model.predict(input_df)[0]

    # Risk Badge
    st.subheader(f"Predicted Stage: {prediction}")
    if prediction == "Normal":
        st.success("Low risk – Continue routine monitoring.")
    elif prediction == "Early":
        st.info("Early stage – Monitor closely and optimise care.")
    elif prediction == "Middle":
        st.warning("Moderate stage – Refer to nephrology.")
    else:
        st.error("Advanced stage – Immediate specialist intervention required.")

    # Compact Speedometer Gauge
    st.markdown("### eGFR Gauge")
    fig, ax = plt.subplots(figsize=(4, 2))  # compact size
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    # Define zones
    zones = [0, 30, 60, 90, 120]
    colors = ['#de2d26', '#fc9272', '#fdae6b', '#a1d99b']

    # Draw colored wedges
    for i in range(len(colors)):
        theta1 = 180 * (zones[i] / 120)
        theta2 = 180 * (zones[i+1] / 120)
        wedge = patches.Wedge(center=(0, 0), r=1, theta1=theta1, theta2=theta2,
                              facecolor=colors[i], alpha=0.7)
        ax.add_patch(wedge)

    # Draw ticks & labels
    for val in zones:
        angle = np.pi * val / 120
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0.9*x, x], [0.9*y, y], color='black', linewidth=0.8)
        ax.text(1.15*x, 1.15*y, str(val), ha='center', va='center', fontsize=8)

    # Draw needle
    needle_angle = np.pi * min(max(egfr, 0), 120) / 120
    needle_x = np.cos(needle_angle)
    needle_y = np.sin(needle_angle)
    ax.plot([0, needle_x], [0, needle_y], color='blue', linewidth=2)

    # Center circle & label
    ax.add_patch(plt.Circle((0, 0), 0.04, color='black'))
    ax.text(0, -0.15, f'eGFR: {egfr}', ha='center', va='center', fontsize=10, weight='bold')
    ax.set_title('eGFR Risk Gauge', fontsize=12, weight='bold', pad=10)

    st.pyplot(fig)

    # Patient Summary Table 
    st.markdown("### Patient Summary")
    summary_df = pd.DataFrame({
        "Parameter": ["Age", "Sex", "Height (cm)", "Weight (kg)", "eGFR"],
        "Value": [age, sex, height, weight, egfr]
    })
    st.table(summary_df)


# BATCH VIEW TAB 
with tab2:
    st.markdown("## Batch View: Ward/Clinic Patient List")

    # Sample patient data
    batch_data = pd.DataFrame({
        "AGE": [5, 12, 8, 16, 10],
        "SEX": ["Male", "Female", "Female", "Male", "Male"],
        "HEIGHT": [110, 150, 130, 160, 140],
        "WEIGHT": [20, 45, 30, 55, 35],
        "EGFR": [95, 72, 45, 28, 105]
    })

    # Prepare features for model
    batch_features = batch_data.copy()
    batch_features["SEX_0"] = (batch_features["SEX"] == "Male").astype(int)
    batch_features["SEX_1"] = (batch_features["SEX"] == "Female").astype(int)
    batch_features = batch_features[["AGE", "HEIGHT", "WEIGHT", "EGFR", "SEX_0", "SEX_1"]]

    # Predict stages
    batch_data["Predicted Stage"] = model.predict(batch_features)

    # Function to color-code stage
    def stage_badge(stage):
        if stage == "Normal":
            return "Normal"
        elif stage == "Early":
            return "Early"
        elif stage == "Middle":
            return "Middle"
        else:
            return "End"

    batch_data["Risk"] = batch_data["Predicted Stage"].apply(stage_badge)

    # Display table
    st.dataframe(batch_data[["AGE", "SEX", "HEIGHT", "WEIGHT", "EGFR", "Risk"]],
                 use_container_width=True)