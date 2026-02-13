import streamlit as st
import joblib
import numpy as np

# Load Model and Scaler
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("salary_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

st.title("💰 Salary Prediction App")
st.write("Predict if income is >50K or <=50K based on user details.")

if model is None or scaler is None:
    st.error("Model or Scaler not found! Please run 'train.py' first.")
else:
    # Input Fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=30)
        education_num = st.number_input("Education Num (e.g., 10-13)", min_value=1, max_value=16, value=10)
        
    with col2:
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        hours_per_week = st.number_input("Hours Per Week", min_value=1, max_value=100, value=40)
        
    if st.button("Predict 🚀"):
        # Preprocess Input
        input_data = np.array([[age, education_num, capital_gain, hours_per_week]])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        # Display Result
        if prediction[0] == 1:
            st.success("✅ Prediction: Income >50K")
            st.balloons()
        else:
            st.error("❌ Prediction: Income <=50K")
