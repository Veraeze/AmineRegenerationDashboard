import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ------------------ Load Model and Data ------------------ #

# Load pre-trained Random Forest model
with open('model/rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load pre-cleaned dataset (for EDA visuals)
df = pd.read_csv('data/amine_gen_data_cleaned.csv')

# ------------------ Streamlit App ------------------ #
st.set_page_config(page_title="Amine Regeneration Dashboard", layout="wide")

st.title("📊 Amine Regeneration Dashboard — Objective 2")
st.markdown("Predict optimal reboiler duty and explore key insights from the process dataset.")

# ------------------ Tabs ------------------ #
tabs = st.tabs(["EDA Insights", "Predict Reboiler Duty"])

# ------------------ EDA Tab ------------------ #
with tabs[0]:
    st.header("📈 Exploratory Data Analysis")

    st.subheader("Feature Importance")
    fig1, ax1 = plt.subplots(figsize=(8,5))
    importance = pd.read_csv('data/feature_importance.csv')
    ax1.barh(importance['Feature'], importance['Importance'], color='teal')
    ax1.set_title('Feature Importance')
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Target Distribution (Reboiler Duty)")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.histplot(df['stripper - Spec Value (Duty)'], bins=40, kde=True, ax=ax3)
    st.pyplot(fig3)

# ------------------ Prediction Tab ------------------ #
with tabs[1]:
    st.header("🤖 Predict Reboiler Duty")

    st.markdown("Enter process values below to predict optimal reboiler duty (Btu/hr):")

    # Collect user inputs
    dea7_h2s = st.number_input("DEA7 - H2S Composition (ppm)", min_value=60.0, max_value=110.0, value=80.0)
    dea7_co2 = st.number_input("DEA7 - CO2 Composition (mole %)", min_value=0.012, max_value=0.022, value=0.016)
    dea6_temp = st.number_input("DEA6 - Temperature (°F)", min_value=200, max_value=295, value=210)
    dea6_acid_temp = st.number_input("DEA6 - Acid Gas Temperature (°F)", min_value=200, max_value=295, value=210)
    stripper_condenser_pressure = st.number_input("Stripper - Condenser Pressure (psig)", min_value=19.0, max_value=40.0, value=20.3)
    stripper_reboiler_pressure = st.number_input("Stripper - Reboiler Pressure (psig)", min_value=22.0, max_value=43.0, value=23.3)

    dea7_temp = st.number_input("DEA7 - Temperature (°F)", min_value=266.0, max_value=295.0, value=268.3)
    dea6_co2 = st.number_input("DEA6 - CO2 Composition (mole %)", min_value=0.03972, max_value=0.03972, value=0.03972)
    dea6_h2s = st.number_input("DEA6 - H2S Composition (ppm)", min_value=171.7, max_value=171.7, value=171.7)

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
    'stripper - Stage Pressure (Condenser)': [stripper_condenser_pressure],
    'stripper - Stage Pressure (Reboiler)': [stripper_reboiler_pressure],
    'DEA6 - Temperature': [dea6_temp],
    'DEA6 - CO2 Composition(mole %)': [dea6_co2],
    'DEA6 - H2S Composition(ppm)': [dea6_h2s],
    'DEA6 - Acid Gas Temperature': [dea6_acid_temp],
    'DEA7 - Temperature': [dea7_temp],
    'DEA7 - CO2 Composition(mole %)': [dea7_co2],
    'DEA7 - H2S Composition(ppm)': [dea7_h2s]
})

    # Predict and display result
    if st.button("Predict Reboiler Duty"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Reboiler Duty: {prediction:,.0f} Btu/hr")
