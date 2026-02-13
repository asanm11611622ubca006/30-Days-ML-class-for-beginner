
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load data
try:
    dataset = pd.read_csv('dataset.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Linear Regression
try:
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    lin_pred = lin_reg.predict([[6.5]])
    print(f"Linear Regression Prediction for 6.5: {lin_pred[0]}")
except Exception as e:
    print(f"Error in Linear Regression: {e}")

# Polynomial Regression
try:
    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)
    poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
    print(f"Polynomial Regression Prediction for 6.5: {poly_pred[0]}")
except Exception as e:
    print(f"Error in Polynomial Regression: {e}")

print("Verification complete.")
