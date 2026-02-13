# Verification Script - Titanic Survival Prediction
# This script runs the same code as the notebook to verify everything works

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚢 TITANIC SURVIVAL PREDICTION - VERIFICATION")
print("=" * 60)

# Step 1: Load Data
print("\n📊 Step 1: Loading Data...")
df = pd.read_csv('titanicsurvival.csv')
print(f"   ✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   ✅ Columns: {list(df.columns)}")

# Step 2: Data Cleaning
print("\n🧹 Step 2: Data Cleaning...")
df_clean = df.copy()
df_clean['Age'] = df_clean.groupby(['Pclass', 'Gender'])['Age'].transform(lambda x: x.fillna(x.median()))
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
print(f"   ✅ Missing values handled: {df_clean.isnull().sum().sum()} remaining")

# Step 3: Feature Engineering
print("\n🛠️ Step 3: Feature Engineering...")
df_clean['IsChild'] = (df_clean['Age'] < 18).astype(int)
print("   ✅ Created IsChild feature")

# Step 4: Prepare Features
print("\n🔄 Step 4: Preparing Features...")
X = df_clean[['Pclass', 'Gender', 'Age', 'Fare', 'IsChild']].copy()
y = df_clean['Survived'].copy()

numerical_cols = ['Age', 'Fare', 'IsChild']
categorical_cols = ['Pclass', 'Gender']

# Create preprocessing pipeline
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
print(f"   ✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Step 5: Train Models
print("\n🤖 Step 5: Training Models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB()
}

results = []
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc, 'F1': f1, 'Pipeline': pipeline})
    print(f"   ✅ {name:25} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# Step 6: Cross-Validation
print("\n🔁 Step 6: Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_cv_score = 0
best_cv_model = None

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    if scores.mean() > best_cv_score:
        best_cv_score = scores.mean()
        best_cv_model = name
    print(f"   ✅ {name:25} | CV Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")

print(f"\n   🏆 Best CV Model: {best_cv_model} ({best_cv_score:.4f})")

# Step 7: Hyperparameter Tuning (Random Forest only for speed)
print("\n⚙️ Step 7: Hyperparameter Tuning (Random Forest)...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_params = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}

rf_search = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_search.fit(X_train, y_train)
print(f"   ✅ Best RF CV Score: {rf_search.best_score_:.4f}")
print(f"   ✅ Best Parameters: {rf_search.best_params_}")

# Step 8: Final Evaluation
print("\n📈 Step 8: Final Evaluation...")
best_model = rf_search.best_estimator_
y_pred_final = best_model.predict(X_test)

final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)

print(f"   🎯 Accuracy:  {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"   🎯 Precision: {final_precision:.4f}")
print(f"   🎯 Recall:    {final_recall:.4f}")
print(f"   🎯 F1-Score:  {final_f1:.4f}")

# Step 9: Confusion Matrix
print("\n📊 Step 9: Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")
print(f"   True Positives:  {tp}")

# Step 10: Feature Importance
print("\n🔍 Step 10: Feature Importance...")
feature_names = numerical_cols + list(best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols))
importances = best_model.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
print(importance_df.to_string(index=False))

# Step 11: Custom Prediction
print("\n🔮 Step 11: Custom Prediction Test...")
def predict_survival(Pclass, Gender, Age, Fare):
    IsChild = 1 if Age < 18 else 0
    input_data = pd.DataFrame({
        'Pclass': [Pclass], 'Gender': [Gender.lower()],
        'Age': [Age], 'Fare': [Fare], 'IsChild': [IsChild]
    })
    prediction = best_model.predict(input_data)[0]
    probability = best_model.predict_proba(input_data)[0]
    return prediction, probability[1]

pred, prob = predict_survival(1, 'female', 29, 100)
print(f"   Test: 1st class female, 29yrs, $100 fare")
print(f"   Result: {'Survived ✅' if pred == 1 else 'Not Survived ❌'} | Probability: {prob*100:.2f}%")

pred2, prob2 = predict_survival(3, 'male', 25, 8)
print(f"\n   Test: 3rd class male, 25yrs, $8 fare")
print(f"   Result: {'Survived ✅' if pred2 == 1 else 'Not Survived ❌'} | Probability: {prob2*100:.2f}%")

# Step 12: Save Model
print("\n💾 Step 12: Saving Model...")
joblib.dump(best_model, 'best_titanic_model.pkl')
print("   ✅ Model saved as 'best_titanic_model.pkl'")

print("\n" + "=" * 60)
print("🎉 VERIFICATION COMPLETE - ALL STEPS PASSED!")
print("=" * 60)
