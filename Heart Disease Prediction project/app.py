"""
Heart Disease Prediction - Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

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

print("✅ Model loaded successfully!")

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Create a dictionary for the input
        input_dict = {
            'Age': float(data['age']),
            'Sex': data['sex'],
            'ChestPainType': data['chestPainType'],
            'RestingBP': float(data['restingBP']),
            'Cholesterol': float(data['cholesterol']),
            'FastingBS': int(data['fastingBS']),
            'RestingECG': data['restingECG'],
            'MaxHR': float(data['maxHR']),
            'ExerciseAngina': data['exerciseAngina'],
            'Oldpeak': float(data['oldpeak']),
            'ST_Slope': data['stSlope']
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        
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
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability': {
                'no_disease': float(probability[0] * 100),
                'disease': float(probability[1] * 100)
            },
            'message': 'Heart Disease Detected ⚠️' if prediction == 1 else 'No Heart Disease ✅'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏥 HEART DISEASE PREDICTION WEB APPLICATION")
    print("="*70)
    print("🚀 Starting Flask server...")
    print("📍 Open your browser and navigate to: http://localhost:5000")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
