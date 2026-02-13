# XGBoost Model - Additional Script for Titanic Survival Prediction
# This script adds XGBoost model evaluation to your project

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 XGBOOST MODEL - TITANIC SURVIVAL PREDICTION")
print("=" * 60)

# Load and prepare data
df = pd.read_csv('titanicsurvival.csv')
df_clean = df.copy()
df_clean['Age'] = df_clean.groupby(['Pclass', 'Gender'])['Age'].transform(lambda x: x.fillna(x.median()))
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
df_clean['IsChild'] = (df_clean['Age'] < 18).astype(int)

X = df_clean[['Pclass', 'Gender', 'Age', 'Fare', 'IsChild']].copy()
y = df_clean['Survived'].copy()

numerical_cols = ['Age', 'Fare', 'IsChild']
categorical_cols = ['Pclass', 'Gender']

# Preprocessing
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📊 Dataset: {len(df)} samples")
print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")

# XGBoost Base Model
print("\n🔹 Training XGBoost Base Model...")
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])

xgb_pipeline.fit(X_train, y_train)
y_pred_base = xgb_pipeline.predict(X_test)
base_accuracy = accuracy_score(y_test, y_pred_base)
print(f"   ✅ Base XGBoost Accuracy: {base_accuracy:.4f} ({base_accuracy*100:.2f}%)")

# Cross-Validation
print("\n🔹 Cross-Validation (5-Fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_pipeline, X, y, cv=cv, scoring='accuracy')
print(f"   ✅ CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Hyperparameter Tuning
print("\n🔹 Hyperparameter Tuning with GridSearchCV...")
xgb_tuning_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    xgb_tuning_pipeline, 
    param_grid, 
    cv=3, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n   ✅ Best Parameters: {grid_search.best_params_}")
print(f"   ✅ Best CV Score: {grid_search.best_score_:.4f}")

# Final Evaluation
print("\n🔹 Final Evaluation on Test Set...")
best_xgb = grid_search.best_estimator_
y_pred_tuned = best_xgb.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print(f"\n   🎯 Tuned XGBoost Accuracy: {tuned_accuracy:.4f} ({tuned_accuracy*100:.2f}%)")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=['Not Survived', 'Survived']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
tn, fp, fn, tp = cm.ravel()
print(f"📊 Confusion Matrix:")
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")

# Feature Importance
print("\n🔍 Feature Importance:")
feature_names = numerical_cols + list(best_xgb.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols))
importances = best_xgb.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
print(importance_df.to_string(index=False))

# Save model
joblib.dump(best_xgb, 'xgboost_titanic_model.pkl')
print(f"\n💾 Model saved as 'xgboost_titanic_model.pkl'")

# Test predictions
print("\n🔮 Sample Predictions:")

def predict_with_xgb(Pclass, Gender, Age, Fare):
    IsChild = 1 if Age < 18 else 0
    input_data = pd.DataFrame({
        'Pclass': [Pclass], 'Gender': [Gender.lower()],
        'Age': [Age], 'Fare': [Fare], 'IsChild': [IsChild]
    })
    pred = best_xgb.predict(input_data)[0]
    prob = best_xgb.predict_proba(input_data)[0]
    return pred, prob[1]

# Test cases
test_cases = [
    (1, 'female', 29, 100, "1st class female"),
    (3, 'male', 25, 8, "3rd class male"),
    (2, 'male', 8, 30, "2nd class child"),
]

for pclass, gender, age, fare, desc in test_cases:
    pred, prob = predict_with_xgb(pclass, gender, age, fare)
    result = "Survived ✅" if pred == 1 else "Not Survived ❌"
    print(f"   {desc}: {result} | Probability: {prob*100:.2f}%")

print("\n" + "=" * 60)
print("🎉 XGBOOST ANALYSIS COMPLETE!")
print("=" * 60)
