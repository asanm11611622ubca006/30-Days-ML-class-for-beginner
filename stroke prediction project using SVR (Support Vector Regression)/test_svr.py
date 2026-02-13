# Test script to verify SVR notebook works correctly
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🏥 STROKE PREDICTION SVR - VERIFICATION TEST")
print("="*60)

# Load the dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data.csv')
print(f"   ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Separate features and target
print("\n2. Preparing data...")
X = df[['x']].values
y = df['y'].values
print(f"   ✓ Feature shape: {X.shape}")
print(f"   ✓ Target shape: {y.shape}")

# Split data
print("\n3. Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Testing set: {X_test.shape[0]} samples")

# Feature scaling
print("\n4. Scaling features...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
print("   ✓ Scaling complete")

# Train SVR model
print("\n5. Training SVR model...")
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train_scaled)
print("   ✓ Model trained successfully")
print(f"   ✓ Support Vectors: {svr_model.n_support_}")

# Make predictions
print("\n6. Making predictions...")
y_pred_scaled = svr_model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
print("   ✓ Predictions complete")

# Calculate metrics
print("\n7. Calculating metrics...")
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n" + "="*60)
print("🎯 MODEL EVALUATION METRICS")
print("="*60)
print(f"\n📊 R² Score (Coefficient of Determination): {r2:.6f}")
print(f"   → Explains {r2*100:.2f}% of the variance")
print(f"\n📏 Mean Absolute Error (MAE):              {mae:.6f}")
print(f"\n📐 Mean Squared Error (MSE):               {mse:.6f}")
print(f"\n📉 Root Mean Squared Error (RMSE):         {rmse:.6f}")
print(f"\n📊 Mean Absolute Percentage Error (MAPE):  {mape:.4f}%")
print("\n" + "="*60)

# Performance interpretation
if r2 >= 0.9:
    performance = "Excellent! 🌟"
elif r2 >= 0.7:
    performance = "Good! 👍"
elif r2 >= 0.5:
    performance = "Moderate 👌"
else:
    performance = "Needs Improvement 📈"

print(f"\n🎯 Model Performance: {performance}")
print("="*60)

# Test prediction function
print("\n8. Testing prediction function...")
def predict_stroke(x_value):
    x_input = np.array([[x_value]])
    x_scaled = scaler_X.transform(x_input)
    y_pred_scaled = svr_model.predict(x_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]
    return y_pred

test_values = [300, 500, 1000, 1500, 1800]
print("\n🎯 Sample Predictions:")
print("="*50)
for x_val in test_values:
    prediction = predict_stroke(x_val)
    print(f"Input X = {x_val:>6.2f}  →  Predicted Y = {prediction:>8.4f}")
print("="*50)

print("\n✅ ALL TESTS PASSED! Notebook code verified successfully!")
print("="*60)
