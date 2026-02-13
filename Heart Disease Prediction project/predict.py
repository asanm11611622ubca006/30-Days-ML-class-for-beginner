"""
Heart Disease Prediction - Standalone Prediction Script
"""

import pickle
import numpy as np
import pandas as pd

# Load the trained model and preprocessors
print("Loading model and preprocessors...")
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✅ Model loaded successfully!\n")

def predict_heart_disease(patient_data):
    """
    Make prediction for a single patient
    
    Parameters:
    patient_data: dict with keys:
        Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
        RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    
    Returns:
    prediction: 0 (No disease) or 1 (Disease)
    probability: probability scores for each class
    """
    # Create DataFrame
    input_df = pd.DataFrame([patient_data])
    
    # Encode categorical features
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Ensure column order matches training data
    input_df = input_df[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return prediction, probability

# ============================================================================
# TEST CASES
# ============================================================================

print("="*70)
print("HEART DISEASE PREDICTION - TEST CASES")
print("="*70)

# Test Case 1: Patient without heart disease (from dataset row 1)
print("\n📋 Test Case 1: Patient Profile")
patient1 = {
    'Age': 40,
    'Sex': 'M',
    'ChestPainType': 'ATA',
    'RestingBP': 140,
    'Cholesterol': 289,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 172,
    'ExerciseAngina': 'N',
    'Oldpeak': 0.0,
    'ST_Slope': 'Up'
}

print("Patient Details:")
for key, value in patient1.items():
    print(f"  {key}: {value}")

prediction1, prob1 = predict_heart_disease(patient1)
print(f"\n🔍 Prediction: {'❌ Heart Disease Detected' if prediction1 == 1 else '✅ No Heart Disease'}")
print(f"📊 Confidence:")
print(f"  - No Disease: {prob1[0]*100:.2f}%")
print(f"  - Disease: {prob1[1]*100:.2f}%")

# Test Case 2: Patient with heart disease (from dataset row 2)
print("\n" + "="*70)
print("\n📋 Test Case 2: Patient Profile")
patient2 = {
    'Age': 65,
    'Sex': 'M',
    'ChestPainType': 'ASY',
    'RestingBP': 170,
    'Cholesterol': 263,
    'FastingBS': 1,
    'RestingECG': 'Normal',
    'MaxHR': 112,
    'ExerciseAngina': 'Y',
    'Oldpeak': 2.0,
    'ST_Slope': 'Flat'
}

print("Patient Details:")
for key, value in patient2.items():
    print(f"  {key}: {value}")

prediction2, prob2 = predict_heart_disease(patient2)
print(f"\n🔍 Prediction: {'❌ Heart Disease Detected' if prediction2 == 1 else '✅ No Heart Disease'}")
print(f"📊 Confidence:")
print(f"  - No Disease: {prob2[0]*100:.2f}%")
print(f"  - Disease: {prob2[1]*100:.2f}%")

print("\n" + "="*70)
print("✨ Prediction script completed successfully!")
print("="*70)
