# AmineRegenerationDashboard

An interactive **Streamlit dashboard** for predicting **optimal reboiler duty** in amine regeneration systems. Includes integrated **exploratory data analysis (EDA)** visualizations and **machine learning predictions** based on process simulation data.

---

##  Objectives

- **Objective 2:** Predict optimal reboiler duty using machine learning (Random Forest).
- **EDA Insights:** Visualize key process relationships like correlations, feature importance, and distribution plots.
- **Simple Interface:** User-friendly web app to assist energy optimization in acid gas removal processes.

---

##  Technologies Used

- Python
- Streamlit
- pandas, scikit-learn, seaborn, matplotlib

---

##  Project Structure

reboilerduty/
├── dashboard/
│   ├── app.py
│   ├── data/
│   │   ├── amine_gen_data_cleaned.csv
│   │   └── feature_importance.csv
│   ├── model/
│   │   └── rf_model.pkl
├── requirements.txt
└── README.md