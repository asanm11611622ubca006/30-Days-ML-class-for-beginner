# Student Marks Prediction

This project uses Machine Learning (Decision Tree Regression) to predict student marks based on their study habits and details.

## Project Structure

- **`data.csv`**: The dataset containing student information.
  - Columns: `hours` (study hours), `age`, `internet` (0 or 1), `marks` (target).
- **`student_marks_prediction.ipynb`**: The Jupyter Notebook containing the complete code for data analysis, preprocessing, model training, and evaluation.

## How to Run

1. Ensure you have Python installed with the following libraries:
    - `pandas`
    - `numpy`
    - `matplotlib`
    - `seaborn`
    - `scikit-learn`
2. Open the Jupyter Notebook:

    ```bash
    jupyter notebook student_marks_prediction.ipynb
    ```

3. Run all cells to see the analysis and model predictions.

## Model

The project uses a **Decision Tree Regressor** to predict continuous mark values. It handles missing data in the 'hours' column by filling with the mean.
