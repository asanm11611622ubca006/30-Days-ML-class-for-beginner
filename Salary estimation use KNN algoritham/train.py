import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# 1. Load Dataset
print("Loading dataset...")
files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not files:
    raise FileNotFoundError("No CSV file found in the current directory.")
csv_file = files[0] # Auto-detect
print(f"Detected file: {csv_file}")
df = pd.read_csv(csv_file)

# 2. EDA
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- First 5 Rows ---")
print(df.head())

# 3. Handle Missing Values
# Impute numerical with median, categorical with mode
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# 4. Encode Target
# <=50K -> 0, >50K -> 1
# Check unique values to ensure correct mapping
print(f"\nUnique values in income before encoding: {df['income'].unique()}")
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
print(f"Unique values in income after encoding: {df['income'].unique()}")

# 5. Feature Selection
features = ['age', 'education.num', 'capital.gain', 'hours.per.week']
X = df[features]
y = df['income']

# 6. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Model Training & Comparison
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# KNN Selection
print("\n--- Finding Best K for KNN ---")
k_values = [3, 5, 7, 9]
best_k = 3
best_k_acc = 0
knn_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    knn_accuracies.append(score)
    print(f"K={k}, Accuracy={score:.4f}")
    if score > best_k_acc:
        best_k_acc = score
        best_k = k

print(f"Best K selected: {best_k}")
models[f"KNN (k={best_k})"] = KNeighborsClassifier(n_neighbors=best_k)

# Train and Evaluate
results = {}
best_model_name = ""
best_model_score = 0
best_model_obj = None

print("\n--- Model Evaluation ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Matrix": cm
    }
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    
    if acc > best_model_score:
        best_model_score = acc
        best_model_name = name
        best_model_obj = model

print(f"\n🏆 Best Model: {best_model_name} with Accuracy check {best_model_score:.4f}")

# 9. Saving Artifacts
print("\nSaving best model and scaler...")
joblib.dump(best_model_obj, 'salary_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved 'salary_model.pkl' and 'scaler.pkl'")

# 10. Visualization
# Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Income Class Distribution")
plt.savefig('class_distribution.png')
print("Saved 'class_distribution.png'")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['income']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig('correlation_heatmap.png')
print("Saved 'correlation_heatmap.png'")

# KNN Accuracy Plot
plt.figure(figsize=(6, 4))
plt.plot(k_values, knn_accuracies, marker='o')
plt.title("KNN Accuracy vs K Value")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.savefig('knn_accuracy.png')
print("Saved 'knn_accuracy.png'")

# Confusion Matrix Heatmap (Best Model)
plt.figure(figsize=(6, 5))
sns.heatmap(results[best_model_name]["Matrix"], annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig('confusion_matrix_best.png')
print("Saved 'confusion_matrix_best.png'")

print("\nProcessing Complete!")
