# 🚢 Titanic Survival Prediction - ML Project

A comprehensive Machine Learning project to predict Titanic passenger survival using multiple algorithms, feature engineering, and hyperparameter tuning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)

---

## 📊 Project Overview

This project achieves **83.80% accuracy** in predicting Titanic passenger survival using Random Forest classifier with proper preprocessing, feature engineering, and hyperparameter tuning.

### 🎯 Key Features

- ✅ Data exploration and visualization
- ✅ Missing value imputation (grouped median)
- ✅ Feature engineering (AgeBand, FareBand, IsChild)
- ✅ Preprocessing pipeline (no data leakage)
- ✅ 8 ML models compared
- ✅ Cross-validation with StratifiedKFold
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Interactive Streamlit web app
- ✅ Custom prediction function

---

## 📁 Project Structure

```
📦 Titanic Survival Prediction
├── 📓 Titanic_Survival_Prediction.ipynb  # Main Jupyter Notebook
├── 🌐 app.py                              # Streamlit Web App
├── 🚀 xgboost_model.py                    # XGBoost model script
├── 🔍 run_verification.py                 # Verification script
├── 💾 best_titanic_model.pkl              # Saved Random Forest model
├── 💾 xgboost_titanic_model.pkl           # Saved XGBoost model
├── 📋 requirements.txt                    # Dependencies
├── 📊 titanicsurvival.csv                 # Dataset
└── 📖 README.md                           # This file
```

---

## 🏆 Model Performance

| Model | Test Accuracy | CV Mean Accuracy |
|-------|--------------|------------------|
| 🥇 **Random Forest** | **83.80%** | 82.38% |
| 🥈 Gradient Boosting | 78.77% | 83.61% |
| 🥉 XGBoost | 81.01% | 81.93% |
| K-Nearest Neighbors | 79.89% | 82.15% |
| Decision Tree | 79.89% | 78.23% |
| Logistic Regression | 78.77% | 80.02% |
| SVM | 78.77% | 79.57% |
| Naive Bayes | 76.54% | 78.45% |

---

## 🔧 Installation

1. **Clone or download this project**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run Jupyter Notebook:**
```bash
jupyter notebook Titanic_Survival_Prediction.ipynb
```

4. **Or run Streamlit App:**
```bash
streamlit run app.py
```

---

## 🚀 Quick Start

### Option 1: Jupyter Notebook
Open `Titanic_Survival_Prediction.ipynb` and run cells block-by-block.

### Option 2: Streamlit Web App
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### Option 3: Python Script
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('best_titanic_model.pkl')

# Predict
input_data = pd.DataFrame({
    'Pclass': [1],
    'Gender': ['female'],
    'Age': [29],
    'Fare': [100],
    'IsChild': [0]
})

prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

print(f"Prediction: {'Survived' if prediction[0] == 1 else 'Not Survived'}")
print(f"Survival Probability: {probability[0][1]*100:.2f}%")
```

---

## 📊 Dataset

| Column | Description |
|--------|-------------|
| Pclass | Passenger class (1, 2, 3) |
| Gender | male/female |
| Age | Age in years |
| Fare | Ticket fare |
| Survived | Target (0 = No, 1 = Yes) |

**Dataset Stats:**
- Total passengers: 891
- Survival rate: 38.4%
- Missing values: Age (19.9%)

---

## 🔍 Key Insights

1. **Gender**: Females had 74% survival rate vs. 19% for males
2. **Class**: 1st class passengers had better survival rates
3. **Age**: Children were prioritized for rescue
4. **Fare**: Higher fare correlated with better survival

### Feature Importance (Random Forest)

| Feature | Importance |
|---------|------------|
| Fare | 28.6% |
| Age | 22.1% |
| Gender (male) | 17.8% |
| Gender (female) | 16.8% |
| Pclass | 12.3% |
| IsChild | 2.5% |

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-Learn** - ML models and preprocessing
- **XGBoost** - Gradient boosting
- **Matplotlib & Seaborn** - Visualization
- **Streamlit** - Web application
- **Joblib** - Model serialization

---

## 📝 Notebook Contents

1. **Data Loading & Exploration** - Load, inspect, visualize
2. **Data Cleaning** - Handle missing values
3. **Feature Engineering** - Create new features
4. **Preprocessing Pipeline** - ColumnTransformer, no leakage
5. **Model Training** - 7 ML algorithms
6. **Cross-Validation** - StratifiedKFold (5-fold)
7. **Hyperparameter Tuning** - GridSearchCV
8. **Final Evaluation** - Metrics, confusion matrix
9. **Feature Importance** - Visualization
10. **Custom Prediction** - Interactive function
11. **Model Saving** - Joblib export

---

## 📧 Usage Example

```python
# Predict survival for a passenger
predict_survival(
    Pclass=1,       # 1st, 2nd, or 3rd class
    Gender='female', # 'male' or 'female'
    Age=29,         # Age in years
    Fare=100        # Ticket fare in $
)

# Output:
# 🎯 Prediction: Survived ✅
# 📊 Probability of Survival: 99.00%
```

---

## 📜 License

This project is for educational purposes.

---

**Created with ❤️ using Python & Machine Learning**
