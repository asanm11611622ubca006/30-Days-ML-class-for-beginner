# Car Price Prediction Using Random Forest Regression
# This script implements the complete ML pipeline

# 1. Import Libraries
print("="*70)
print("🚗 CAR PRICE PREDICTION - RANDOM FOREST REGRESSION")
print("="*70)
print("\n📦 Importing libraries...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!\n")

# 2. Load Dataset
print("="*70)
print("📊 LOADING DATASET")
print("="*70)

df = pd.read_csv('dataset.csv')
print(f"\n✅ Dataset loaded successfully!")
print(f"   - Total Cars: {df.shape[0]}")
print(f"   - Total Features: {df.shape[1]}")
print(f"\n📋 First 3 rows:\n")
print(df.head(3))

# Check for missing values
print(f"\n❓ Missing Values Check:")
missing = df.isnull().sum().sum()
if missing == 0:
    print("   ✅ No missing values found!")
else:
    print(f"   ⚠️ Found {missing} missing values")

# Target variable statistics
print(f"\n💰 Target Variable (Price) Statistics:")
print(f"   - Mean Price: ${df['price'].mean():,.2f}")
print(f"   - Median Price: ${df['price'].median():,.2f}")
print(f"   - Min Price: ${df['price'].min():,.2f}")
print(f"   - Max Price: ${df['price'].max():,.2f}")

# 3. Data Preprocessing
print("\n" + "="*70)
print("🔧 DATA PREPROCESSING")
print("="*70)

data = df.copy()

# Drop non-predictive columns
data = data.drop(columns=['car_ID', 'CarName'])
print(f"\n✅ Dropped non-predictive columns: car_ID, CarName")
print(f"   - Remaining features: {data.shape[1] - 1} (excluding target)")

# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('price')

print(f"\n🔤 Categorical Features: {len(categorical_cols)}")
print(f"🔢 Numerical Features: {len(numerical_cols)}")

# Encode categorical variables
print(f"\n🔄 Encoding categorical variables...")
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print(f"✅ Encoding complete!")
print(f"   - Features after encoding: {data_encoded.shape[1] - 1}")

# Separate features and target
X = data_encoded.drop('price', axis=1)
y = data_encoded['price']
print(f"\n✅ Features (X) shape: {X.shape}")
print(f"✅ Target (y) shape: {y.shape}")

# 4. Train-Test Split
print("\n" + "="*70)
print("✂️ TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data split completed:")
print(f"   - Training samples: {X_train.shape[0]} ({(X_train.shape[0]/len(X))*100:.1f}%)")
print(f"   - Testing samples: {X_test.shape[0]} ({(X_test.shape[0]/len(X))*100:.1f}%)")
print(f"   - Total features: {X_train.shape[1]}")

# 5. Feature Scaling
print("\n" + "="*70)
print("📊 FEATURE SCALING")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Feature scaling completed using StandardScaler")

# 6. Train Random Forest Model
print("\n" + "="*70)
print("🌲 TRAINING RANDOM FOREST REGRESSION MODEL")
print("="*70)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print(f"\n⏳ Training model with 100 trees...")
rf_model.fit(X_train_scaled, y_train)
print(f"✅ Model training completed successfully!")

# 7. Make Predictions
print("\n" + "="*70)
print("🔮 MAKING PREDICTIONS")
print("="*70)

y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

print(f"\n✅ Predictions completed!")

# 8. Calculate R² Score and Metrics
print("\n" + "="*70)
print("🎯 MODEL EVALUATION - R² SCORE")
print("="*70)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n" + "🌟"*30)
print(f"\n   📊 R² SCORE (PRIMARY METRIC)")
print(f"\n      Training R² Score:   {r2_train:.4f} ({r2_train*100:.2f}%)")
print(f"      Testing R² Score:    {r2_test:.4f} ({r2_test*100:.2f}%) ⭐")
print("\n" + "🌟"*30)

# Interpretation
if r2_test >= 0.9:
    interpretation = "🌟 EXCELLENT - Model explains over 90% of variance!"
elif r2_test >= 0.8:
    interpretation = "✨ VERY GOOD - Strong predictive performance!"
elif r2_test >= 0.7:
    interpretation = "✅ GOOD - Decent predictive capability!"
elif r2_test >= 0.6:
    interpretation = "⚠️ FAIR - Moderate predictive performance"
else:
    interpretation = "❌ POOR - Model needs improvement"

print(f"\n📊 Interpretation: {interpretation}")

# Additional Metrics
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

print("\n" + "="*70)
print("📈 COMPREHENSIVE PERFORMANCE METRICS")
print("="*70)
print(f"\n{'Metric':<35} {'Training':<15} {'Testing'}")
print("-"*70)
print(f"{'R² Score':<35} {r2_train:<15.4f} {r2_test:.4f}")
print(f"{'MAE (Mean Absolute Error)':<35} ${mae_train:<14,.2f} ${mae_test:,.2f}")
print(f"{'MSE (Mean Squared Error)':<35} ${mse_train:<14,.2f} ${mse_test:,.2f}")
print(f"{'RMSE (Root Mean Squared Error)':<35} ${rmse_train:<14,.2f} ${rmse_test:,.2f}")

# Overfitting check
overfitting_gap = r2_train - r2_test
print(f"\n🔍 Overfitting Check:")
print(f"   - R² Gap (Train - Test): {overfitting_gap:.4f}")
if overfitting_gap < 0.05:
    print("   - Status: ✅ Model generalizes well!")
elif overfitting_gap < 0.1:
    print("   - Status: ⚠️ Slight overfitting")
else:
    print("   - Status: ❌ Significant overfitting detected")

# 9. Sample Predictions
print("\n" + "="*70)
print("🎯 SAMPLE PREDICTIONS (First 10 Test Cases)")
print("="*70)

sample_results = pd.DataFrame({
    'Actual Price': y_test.values[:10],
    'Predicted Price': y_pred_test[:10],
    'Difference': y_test.values[:10] - y_pred_test[:10],
    'Error %': np.abs((y_test.values[:10] - y_pred_test[:10]) / y_test.values[:10] * 100)
})

print("\n" + sample_results.to_string(index=False))

# 10. Feature Importance (Top 10)
print("\n" + "="*70)
print("⭐ TOP 10 MOST IMPORTANT FEATURES")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

print()
for idx, row in feature_importance.iterrows():
    print(f"   {row['Feature']:<30} {row['Importance']:.4f}")

# 11. Visualizations
print("\n" + "="*70)
print("📊 GENERATING VISUALIZATIONS")
print("="*70)

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('🚗 Car Price Prediction Model - Performance Dashboard', fontsize=18, fontweight='bold')

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_title(f'Actual vs Predicted Prices\nR² = {r2_test:.4f}', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residual Plot
residuals = y_test - y_pred_test
axes[0, 1].scatter(y_pred_test, residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.5, color='orange')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Residual Plot\n(Errors Distribution)', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance (Top 15)
top_features = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[1, 0].barh(range(len(top_features)), top_features['importance'], color='skyblue', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'], fontsize=9)
axes[1, 0].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Top 15 Feature Importance', fontsize=13, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, axis='x', alpha=0.3)

# 4. Error Distribution
axes[1, 1].hist(residuals, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
axes[1, 1].set_xlabel('Prediction Error ($)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'Error Distribution\nMean Error = ${residuals.mean():.2f}', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'model_performance_dashboard.png'")

# 12. Final Summary
print("\n" + "🌟"*35)
print("\n" + " "*20 + "🏆 FINAL SUMMARY")
print("\n" + "🌟"*35)
print(f"\n📊 Dataset Information:")
print(f"   • Total Cars: {df.shape[0]}")
print(f"   • Features Used: {X.shape[1]}")
print(f"   • Training Samples: {X_train.shape[0]}")
print(f"   • Testing Samples: {X_test.shape[0]}")
print(f"\n🤖 Model: Random Forest Regression")
print(f"   • Number of Trees: {rf_model.n_estimators}")
print(f"   • Max Depth: {rf_model.max_depth}")
print(f"\n🎯 PRIMARY METRIC - R² SCORE:")
print(f"   • Training R²:  {r2_train:.4f} ({r2_train*100:.2f}%)")
print(f"   • Testing R²:   {r2_test:.4f} ({r2_test*100:.2f}%) ⭐⭐⭐")
print(f"\n📈 Other Performance Metrics:")
print(f"   • MAE (Test):   ${mae_test:,.2f}")
print(f"   • RMSE (Test):  ${rmse_test:,.2f}")
print(f"\n💡 What R² Score Means:")
print(f"   The model explains {r2_test*100:.2f}% of the variance in car prices.")
print(f"   {interpretation}")
print("\n" + "="*70)
print("✅ PROJECT COMPLETED SUCCESSFULLY!")
print("="*70)

plt.show()
