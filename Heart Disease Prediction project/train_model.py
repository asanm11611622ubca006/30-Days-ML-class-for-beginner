"""
Heart Disease Prediction - Model Training Script
Using Gaussian Naive Bayes Algorithm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report, 
                            roc_curve, auc, roc_auc_score)
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("HEART DISEASE PREDICTION - MODEL TRAINING")
print("Algorithm: Gaussian Naive Bayes")
print("="*70)

# ============================================================================
# 1. DATA LOADING & EXPLORATION
# ============================================================================
print("\n📊 STEP 1: Loading Dataset...")
df = pd.read_csv('heart.csv')

print(f"\n✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Dataset Info:")
print(f"   Columns: {list(df.columns)}")
print(f"\n   Data Types:")
print(df.dtypes)

print(f"\n🔍 First 5 Records:")
print(df.head())

print(f"\n📈 Statistical Summary:")
print(df.describe())

print(f"\n🔎 Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "   ✅ No missing values found!")

print(f"\n🔎 Duplicate Records:")
duplicates = df.duplicated().sum()
print(f"   Total Duplicates: {duplicates}")

print(f"\n🎯 Target Distribution:")
target_counts = df['HeartDisease'].value_counts()
print(target_counts)
print(f"   No Disease (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"   Disease (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n📊 STEP 2: Performing Exploratory Data Analysis...")

# Target Distribution Plot
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
target_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Heart Disease Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Heart Disease (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Age Distribution
plt.subplot(2, 3, 2)
plt.hist(df['Age'], bins=20, color='#3498db', edgecolor='black')
plt.title('Age Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Cholesterol Distribution
plt.subplot(2, 3, 3)
plt.hist(df['Cholesterol'], bins=20, color='#9b59b6', edgecolor='black')
plt.title('Cholesterol Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')

# Heart Disease by Sex
plt.subplot(2, 3, 4)
pd.crosstab(df['Sex'], df['HeartDisease']).plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Heart Disease by Sex', fontsize=14, fontweight='bold')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['No Disease', 'Disease'])

# Heart Disease by Chest Pain Type
plt.subplot(2, 3, 5)
pd.crosstab(df['ChestPainType'], df['HeartDisease']).plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Heart Disease by Chest Pain Type', fontsize=14, fontweight='bold')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])

# MaxHR vs Age colored by Heart Disease
plt.subplot(2, 3, 6)
for disease in [0, 1]:
    subset = df[df['HeartDisease'] == disease]
    plt.scatter(subset['Age'], subset['MaxHR'], 
               label=f"{'No Disease' if disease == 0 else 'Disease'}", 
               alpha=0.5)
plt.title('Max Heart Rate vs Age', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.legend()

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
print("   ✅ EDA plots saved as 'eda_plots.png'")

# Correlation Heatmap for Numerical Features
print("\n📊 Creating Correlation Heatmap...")
plt.figure(figsize=(12, 8))

# Select numerical columns
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease']
correlation = df[numerical_cols].corr()

sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Numerical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✅ Correlation heatmap saved as 'correlation_heatmap.png'")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n🔧 STEP 3: Data Preprocessing...")

# Make a copy for preprocessing
df_processed = df.copy()

# Encode categorical features
print("   📝 Encoding categorical features...")
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"      - {col}: {list(le.classes_)}")

# Separate features and target
X = df_processed.drop('HeartDisease', axis=1)
y = df_processed['HeartDisease']

print(f"\n   ✅ Features shape: {X.shape}")
print(f"   ✅ Target shape: {y.shape}")

# Split data
print("\n   📊 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"      - Training set: {X_train.shape[0]} samples")
print(f"      - Testing set: {X_test.shape[0]} samples")

# Feature Scaling
print("\n   ⚖️  Applying feature scaling (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Feature scaling completed!")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================
print("\n🤖 STEP 4: Training Gaussian Naive Bayes Model...")

model = GaussianNB()
model.fit(X_train_scaled, y_train)

print("   ✅ Model training completed!")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n📈 STEP 5: Model Evaluation...")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*70)
print("📊 MODEL PERFORMANCE METRICS")
print("="*70)
print(f"✅ Accuracy:  {accuracy*100:.2f}%")
print(f"✅ Precision: {precision*100:.2f}%")
print(f"✅ Recall:    {recall*100:.2f}%")
print(f"✅ F1-Score:  {f1*100:.2f}%")
print("="*70)

# Detailed Classification Report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# Confusion Matrix
print("\n📊 Creating Evaluation Visualizations...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15, 5))

# Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Metrics Comparison
plt.subplot(1, 3, 3)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy*100, precision*100, recall*100, f1*100]
colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
bars = plt.bar(metrics_names, metrics_values, color=colors, edgecolor='black')
plt.ylim([0, 100])
plt.ylabel('Score (%)')
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("   ✅ Evaluation plots saved as 'model_evaluation.png'")

# ============================================================================
# 6. SAVE MODEL AND PREPROCESSORS
# ============================================================================
print("\n💾 STEP 6: Saving Model and Preprocessors...")

# Save the model
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✅ Model saved as 'heart_disease_model.pkl'")

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✅ Scaler saved as 'scaler.pkl'")

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("   ✅ Label encoders saved as 'label_encoders.pkl'")

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print("   ✅ Feature names saved as 'feature_names.pkl'")

print("\n" + "="*70)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n📁 Generated Files:")
print("   1. heart_disease_model.pkl - Trained Naive Bayes model")
print("   2. scaler.pkl - StandardScaler for feature scaling")
print("   3. label_encoders.pkl - Label encoders for categorical features")
print("   4. feature_names.pkl - Feature column names")
print("   5. eda_plots.png - Exploratory data analysis visualizations")
print("   6. correlation_heatmap.png - Feature correlation heatmap")
print("   7. model_evaluation.png - Model performance visualizations")
print("\n✨ Ready to make predictions!")
print("="*70)
