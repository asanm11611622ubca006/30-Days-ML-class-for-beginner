import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# Handwritten Digit Recognition using Random Forest
This notebook implements a Random Forest Classifier to recognize handwritten digits from the MNIST dataset.
It includes data exploration, visualization, model training, evaluation, and a custom prediction function.
"""),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image, ImageOps
import os

# Configure visualization
%matplotlib inline
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
"""),

    nbf.v4.new_markdown_cell("""## 1. Data Loading"""),
    nbf.v4.new_code_cell("""# Load the dataset
# As per instructions, the file is in the same folder
try:
    df = pd.read_csv('digit.csv')
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'digit.csv' not found. Please ensure the file is in the same directory.")
"""),

    nbf.v4.new_markdown_cell("""## 2. Data Exploration & Visualization"""),
    nbf.v4.new_code_cell("""# Display first few rows
df.head()
"""),

    nbf.v4.new_markdown_cell("""### Target Distribution
Checking for class imbalance.
"""),
    nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Distribution of Digits (0-9)')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.show()
"""),

    nbf.v4.new_markdown_cell("""### Sample Digits Visualization
Let's look at some random samples from the dataset.
"""),
    nbf.v4.new_code_cell("""def plot_sample_digits(data, labels, n=25):
    plt.figure(figsize=(10, 10))
    indices = np.random.randint(0, len(data), n)
    
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        # Pixels are from column 1 onwards (column 0 is label)
        image = data.iloc[idx].values.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {labels.iloc[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Separate features and label
X = df.drop('label', axis=1)
y = df['label']

plot_sample_digits(X, y)
"""),

    nbf.v4.new_markdown_cell("""## 3. Data Preprocessing"""),
    nbf.v4.new_code_cell("""# Normalize pixel values (0-255 -> 0-1)
# This helps the model converge faster (though Random Forest is robust to scaling, it's good practice)
X = X / 255.0

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")
"""),

    nbf.v4.new_markdown_cell("""## 4. Model Training (Random Forest)"""),
    nbf.v4.new_code_cell("""# Initialize Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
print("Training the model... This may take a moment.")
clf.fit(X_train, y_train)
print("Model created and trained!")
"""),

    nbf.v4.new_markdown_cell("""## 5. Evaluation"""),
    nbf.v4.new_code_cell("""# Make predictions
y_pred = clf.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
"""),

    nbf.v4.new_markdown_cell("""### Confusion Matrix Heatmap"""),
    nbf.v4.new_code_cell("""# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
"""),

    nbf.v4.new_markdown_cell("""### Classification Report"""),
    nbf.v4.new_code_cell("""print(classification_report(y_test, y_pred))
"""),

    nbf.v4.new_markdown_cell("""## 6. Feature Importance Visualization
Seeing which pixels are most important for distinguishing digits.
"""),
    nbf.v4.new_code_cell("""importances = clf.feature_importances_
# Reshape to 28x28 image
importance_img = importances.reshape(28, 28)

plt.figure(figsize=(10, 8))
sns.heatmap(importance_img, cmap='hot', square=True)
plt.title('Pixel Importance Heatmap')
plt.axis('off')
plt.show()
"""),

    nbf.v4.new_markdown_cell("""## 7. Misclassified Examples
Let's see where the model made errors.
"""),
    nbf.v4.new_code_cell("""# Find indices where predictions didn't match ground truth
misclassified_indices = np.where(y_pred != y_test)[0]

plt.figure(figsize=(12, 12))
for i, idx in enumerate(misclassified_indices[:20]): # Show first 20 errors
    plt.subplot(4, 5, i + 1)
    
    # We need to access X_test by index (iloc for pandas)
    image = X_test.iloc[idx].values.reshape(28, 28)
    true_label = y_test.iloc[idx]
    pred_label = y_pred[idx]
    
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_label}\\nPred: {pred_label}", color='red')
    plt.axis('off')

plt.tight_layout()
plt.show()
"""),

    nbf.v4.new_markdown_cell("""## 8. Custom Prediction Interface
Predict on your own handwritten digit image.
"""),
    nbf.v4.new_code_cell("""def predict_custom_image(image_path, model):
    try:
        # Load image
        img = Image.open(image_path).convert('L') # Convert to grayscale
        
        # Invert if the image is white background/black digit (MNIST is black background/white digit)
        # We assume typical drawing apps save white background.
        # Check corner pixel; if bright, invert.
        if np.mean(img) > 127: 
             img = ImageOps.invert(img)
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Display the processed image
        plt.figure(figsize=(3,3))
        plt.imshow(img_array, cmap='gray')
        plt.title('Processed Input')
        plt.axis('off')
        plt.show()
        
        # Flatten for prediction (1 row, 784 columns)
        img_flat = img_array.reshape(1, 784)
        
        # Predict
        prediction = model.predict(img_flat)[0]
        confidence = np.max(model.predict_proba(img_flat))
        
        print(f"\\n>>> Predicted Digit: {prediction} (Confidence: {confidence:.2f})")
        return prediction
        
    except Exception as e:
        print(f"Error processing image: {e}")

# Example Usage:
# Create a dummy image for demonstration if one doesn't exist
if not os.path.exists('test_digit.png'):
    # Create a simple image (e.g. a black image) just to prevent error in example
    # In real usage, you would upload a drawing
    dummy_img = Image.new('L', (28, 28), color=0)
    dummy_img.save('test_digit.png')

print("Example prediction on 'test_digit.png':")
predict_custom_image('test_digit.png', clf)

print("\\nTo use this with your own image:")
print("1. Create a 28x28 image (or any size) with a digit.")
print("2. Save it as 'my_digit.png' in this folder.")
print("3. Run: predict_custom_image('my_digit.png', clf)")
""")
]

with open('digit_recognition.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
