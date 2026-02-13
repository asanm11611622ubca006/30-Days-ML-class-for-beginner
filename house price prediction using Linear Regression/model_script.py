import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():
    try:
        # 1. Load Data
        print("Loading data...")
        df = pd.read_csv('dataset.csv')
        print(f"Data loaded. Shape: {df.shape}")

        # 2. Train Model
        print("Training Linear Regression model...")
        X = df[['area']]
        y = df['price']
        
        model = LinearRegression()
        model.fit(X, y)
        
        print(f"Model Trained. Coefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}")

        # 3. Prediction Loop
        while True:
            user_input = input("\nEnter area (sq.ft) to predict price (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            try:
                area = int(user_input)
                input_data = pd.DataFrame([[area]], columns=['area'])
                predicted_price = model.predict(input_data)[0]
                print(f"Predicted Price: {predicted_price:.2f}")
            except ValueError:
                print("Invalid input. Please enter a number.")

    except FileNotFoundError:
        print("Error: 'dataset.csv' not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
