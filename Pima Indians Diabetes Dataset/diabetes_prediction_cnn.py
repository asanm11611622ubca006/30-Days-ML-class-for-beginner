import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import model_from_json
import os

# Set seed for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def load_data(filepath):
    print("Loading data...")
    header_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    # Load dataset
    dataset = pd.read_csv(filepath, names=header_names)
    print(f"Data loaded. Shape: {dataset.shape}")
    return dataset

def preprocess_data(dataset):
    print("Preprocessing data...")
    X = dataset.iloc[:, 0:8].values
    y = dataset.iloc[:, 8].values

    # Standardize the data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Reshape for CNN: (samples, features, channels) -> (samples, 8, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    print("Building improved model...")
    model = Sequential()
    # Increased filters to 64
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    # Added Dropout to prevent overfitting
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # Added extra Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    filepath = 'pima-indians-diabetes.csv'
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    # 1. Load Data
    dataset = load_data(filepath)

    # 2. Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(dataset)

    # 3. Build Model
    model = build_model((8, 1))

    # 4. Train Model
    print("Training improved model (200 epochs)...")
    # Increased epochs and batch_size
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

    # 5. Evaluate
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Accuracy on Test Set: {accuracy*100:.2f}%")

    # 6. Save Model
    print("Saving model...")
    # Serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    # Serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved improved model to disk (model.json and model.h5)")

if __name__ == "__main__":
    main()
