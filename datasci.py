# Importation of the libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set the page title and header layout
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------
# 1. CUSTOM UI STYLING (Injected CSS)
# ------------------------------------------------------------------
st.markdown("""
    <style>
    /* Style the main predict button */
    div.stButton > button:first-child {
        # background-color: #008080;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
    }
    /* Style the metric display cards */
    [data-testid="stMetric"] {
        background-color: #008080;
        padding: 12px;
        border-radius: 8px;
        # border-left: 4px solid #008080;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Risk Assessment Tool")
st.write("Enter patient vitals below to get real-time clinical Risk Prediction.")

# Load the pre-trained Model
model = joblib.load('diabetes_model.joblib')
st.sidebar.header("Patient Intake Vitals")

# Interactive Sliders and inputs for the user
glucose = st.sidebar.slider("Glucose Level (mg/dL)", min_value=70, max_value=200, value=120)
bmi = st.sidebar.slider("Body Mass Index (BMI)", min_value=15.0, max_value=50.0, value=25.0)
age = st.sidebar.slider("Age (Years)", min_value=5, max_value=100, value=18)
steps = st.sidebar.number_input("Average Daily Steps", min_value=500, max_value=20000, value=5000)

# Organize inputs into a dataframe matching training features
input_data = pd.DataFrame({
    'Glucose_Level': [glucose],
    'BMI': [bmi],
    'Age': [age],
    'Daily_Steps': [steps]
})

# Display Patient Summary
st.subheader("Patient Vitals Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Glucose", f"{glucose} mg/dL")
col2.metric("BMI", f"{bmi}")
col3.metric("Age", f"{age} yrs")
col4.metric("Daily Steps", f"{steps:,}")

st.markdown("---")

# Predict Button & Output Layout
if st.button("Run Risk Prediction", type="primary"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    # Split output into two side-by-side columns: Results on Left, Visuals on Right
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("🩺 Diagnostic Results")

        # -----------------------------------------------------------
        # 2. DYNAMIC RISK PROGRESS BAR
        # -----------------------------------------------------------
        st.write(f"**Overall Calculated Risk:** `{probability:.1f}%`")
        st.progress(int(probability))

        # Display the results based on Prediction
        if prediction == 1:
            st.error(f"🚨 HIGH RISK OF DIABETES (Risk Score: {probability:.1f}%)")
            st.warning("Recommendation: Schedule immediate fasting plasma glucose (FPG) test and HbA1c Screening.")
        else:
            st.success(f"✅ LOW RISK / NORMAL (Risk Score: {probability:.1f}%)")
            st.info("Recommendation: Maintain regular annual wellness checkups and healthy physical activity.")

    with res_col2:
        # -----------------------------------------------------------
        # 3. FEATURE IMPORTANCE BAR CHART
        # -----------------------------------------------------------
        st.subheader("Model Feature Importance")
        
        feature_names = ['Glucose Level', 'BMI', 'Age', 'Daily Steps']
        importances = model.feature_importances_ * 100

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance (%)': importances
        }).sort_values(by='Importance (%)', ascending=True)

        fig, ax = plt.subplots(figsize=(5, 2.8))
        
        # Color bar green for low risk, red for high risk
        bar_color = '#d32f2f' if prediction == 1 else '#008080'
        bars = ax.barh(importance_df['Feature'], importance_df['Importance (%)'], color=bar_color)
        
        # Annotate bars with percentage values
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=8)

        ax.set_xlim(0, max(importance_df['Importance (%)']) + 15)
        ax.set_xlabel('Influence (%)', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        st.pyplot(fig)