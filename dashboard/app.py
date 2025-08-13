import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ------------------ Path Setup ------------------ #
BASE_DIR = os.path.dirname(__file__)

# ------------------ Load Model and Data ------------------ #
rf_model_path = os.path.join(BASE_DIR, 'model', 'rf_model.pkl')
clf_model_path = os.path.join(BASE_DIR, 'model', 'amine_degradation.pkl')
scaler_path = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
data_path = os.path.join(BASE_DIR, 'data', 'amine_gen_data_cleaned.csv')
importance_path = os.path.join(BASE_DIR, 'data', 'feature_importance.csv')

with open(rf_model_path, 'rb') as file:
    model = pickle.load(file)

with open(clf_model_path, 'rb') as f:
    clf = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load cleaned data and feature importance
df = pd.read_csv(data_path)
importance = pd.read_csv(importance_path)

# ------------------ Streamlit App ------------------ #
st.set_page_config(page_title="Amine Regeneration Dashboard", layout="wide")

st.title(" Amine Regeneration Dashboard")
st.markdown("Predict optimal reboiler duty and health state of the amine system.")

# ------------------ Tabs ------------------ #
tabs = st.tabs(["EDA Insights", "Predict Reboiler Duty", "Classify Health State"])

# ------------------ EDA Tab ------------------ #
with tabs[0]:
    st.header(" Exploratory Data Analysis")

    st.subheader("Cluster Pairplot")
    if 'Health_State' in df.columns:
        selected_features = ['Delta_CO2', 'Delta_H2S', 'Delta_Temp', 'DEA7 - Molar Flow', 'stripper - Spec Value (Duty)', 'Health_State']
        df_pair = df[selected_features].dropna()

        st.markdown("Due to performance, only a random 1000 rows will be plotted.")
        pairplot_fig = sns.pairplot(df_pair.sample(n=1000, random_state=42), hue='Health_State', palette='Set2')
        st.pyplot(pairplot_fig)
        
    st.subheader("Feature Importance")
    fig1, ax1 = plt.subplots(figsize=(8,5))
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

    st.subheader("Cluster Distribution Count")
    if 'Health_State' in df.columns:
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Health_State', hue='Health_State', palette='Set2', ax=ax4)
        if ax4.legend_ is not None:
            ax4.legend_.remove()         
        ax4.set_title("Number of Records per Health State")
        st.pyplot(fig4)

    

# ------------------ Prediction Tab ------------------ #
with tabs[1]:
    st.header(" Predict Reboiler Duty")

    st.markdown("Enter process values below to predict optimal reboiler duty (Btu/hr):")

    # Collect user inputs
    dea7_h2s = st.number_input("DEA7 - H2S Composition (ppm)", min_value=60.0, max_value=110.0, value=80.0)
    dea7_co2 = st.number_input("DEA7 - CO2 Composition (mole %)", min_value=0.012, max_value=0.022, value=0.016)
    dea7_temp = st.number_input("DEA7 - Temperature (°F)", min_value=266.0, max_value=295.0, value=268.3)

    dea6_temp = st.number_input("DEA6 - Temperature (°F)", min_value=200, max_value=295, value=210)
    dea6_acid_temp = st.number_input("DEA6 - Acid Gas Temperature (°F)", min_value=200, max_value=295, value=210)
    dea6_co2 = st.number_input("DEA6 - CO2 Composition (mole %)", min_value=0.03972, max_value=0.03972, value=0.03972)
    dea6_h2s = st.number_input("DEA6 - H2S Composition (ppm)", min_value=171.7, max_value=171.7, value=171.7)

    stripper_condenser_pressure = st.number_input("Stripper - Condenser Pressure (psig)", min_value=19.0, max_value=40.0, value=20.3)
    stripper_reboiler_pressure = st.number_input("Stripper - Reboiler Pressure (psig)", min_value=22.0, max_value=43.0, value=23.3)

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

# ------------------ Classification Tab ------------------ #
with tabs[2]:
    st.header(" Amine Health State")
    st.markdown("Enter process deltas and values to classify the current health state of the amine system:")

    DEFAULTS = {
        "delta_co2": 0.0,
        "delta_h2s": -87.0,
        "delta_temp": 0.0,
        "molar_flow": 10000.0,
        "reboiler_duty": 50000.0
    }

    delta_co2 = st.number_input("Delta CO2", min_value=-0.05, max_value=0.05, value=DEFAULTS["delta_co2"])
    delta_h2s = st.number_input("Delta H2S", min_value=-150.0, max_value=0.0, value=DEFAULTS["delta_h2s"])
    delta_temp = st.number_input("Delta Temp", min_value=-30.0, max_value=30.0, value=DEFAULTS["delta_temp"])
    molar_flow = st.number_input("DEA7 - Molar Flow", min_value=5000.0, max_value=15000.0, value=DEFAULTS["molar_flow"])
    reboiler_duty = st.number_input("Reboiler Duty (Btu/hr)", min_value=10000.0, max_value=100000.0, value=DEFAULTS["reboiler_duty"])

    if st.button("Classify Health State"):
        features = [[
            delta_co2 or DEFAULTS["delta_co2"],
            delta_h2s or DEFAULTS["delta_h2s"],
            delta_temp or DEFAULTS["delta_temp"],
            molar_flow or DEFAULTS["molar_flow"],
            reboiler_duty or DEFAULTS["reboiler_duty"]
        ]]
        features_scaled = scaler.transform(features)
        prediction = clf.predict(features_scaled)[0]
        st.success(f"Predicted Health State: {prediction}")