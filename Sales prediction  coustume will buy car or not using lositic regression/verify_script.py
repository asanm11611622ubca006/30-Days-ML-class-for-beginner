import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

# Set visualization style
sns.set(style="whitegrid")

def load_dataset():
    possible_names = ['car_data.csv', 'Car_Purchase.csv', 'customer_car_buy.csv', 'DigitalAd_dataset.csv']
    
    # Check specific names first
    for name in possible_names:
        if os.path.exists(name):
            print(f"✅ Feature Detected: Loading dataset from '{name}'")
            return pd.read_csv(name)
    
    # Search for any CSV file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if len(csv_files) == 1:
        print(f"✅ Found single CSV file: '{csv_files[0]}'. Loading...")
        return pd.read_csv(csv_files[0])
    elif len(csv_files) > 1:
        print("⚠️ Multiple CSV files found:", csv_files)
        # Simple heuristic: pick the first one or ask user (in a script we'd input, in notebook we pick first for automation)
        print(f"👉 Loading the first one: '{csv_files[0]}'")
        return pd.read_csv(csv_files[0])
    else:
        raise FileNotFoundError("❌ No dataset file found in current directory!")

try:
    df = load_dataset()
    print("\n--- Dataset info ---")
    print(f"Shape: {df.shape}")
except Exception as e:
    print(e)
    exit()

print("\n--- Basic Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicates ---")
duplicates = df.duplicated().sum()
print(f"Total Duplicates: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print("✅ Duplicates removed.")

# Determine Target Column (assuming it is the last column or named 'Status'/'Purchased')
# Based on user description: Status or Purchased = 1/0
target_col = 'Status' if 'Status' in df.columns else df.columns[-1]
print(f"\n🎯 Target Column Detected: '{target_col}'")

# Separate Features and Target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identifiy Numerical and Categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)

# Define Preprocessing Pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Preprocessing
# We fit on train and transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n✅ Data Preprocessing Complete.")
print(f"Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")

# Initialize Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}
best_model = None
best_score = 0
best_model_name = ""

print("TRAINING RESULTS:\n")

for name, model in models.items():
    # Train
    model.fit(X_train_processed, y_train)
    
    # Predict
    y_pred = model.predict(X_test_processed)
    try:
        y_proba = model.predict_proba(X_test_processed)[:, 1]
    except:
        y_proba = [0]*len(y_test) # Handle cases where proba might not be available directly

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
    
    results[name] = {'Accuracy': acc, 'ROC_AUC': roc, 'Model': model}
    
    print(f"🔹 {name} -> Accuracy: {acc:.4f} | ROC AUC: {roc:.4f}")
    
    # Save best model
    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name

print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_score:.4f}")

# Classification Report for Best Model
print(f"--- Classification Report for {best_model_name} ---")
y_pred_best = best_model.predict(X_test_processed)
print(classification_report(y_test, y_pred_best))

# Save the Model and Preprocessor
joblib.dump(best_model, 'best_car_purchase_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("✅ Best model and preprocessor saved successfully!")
