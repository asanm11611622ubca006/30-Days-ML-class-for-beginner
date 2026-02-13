import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

print("Starting notebook execution...")

try:
    # Read the notebook
    with open('Engine_Energy_Prediction.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Create executor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    # Execute the notebook
    print("Executing notebook cells...")
    ep.preprocess(nb, {'metadata': {'path': './'}})
    
    # Save the executed notebook
    with open('Engine_Energy_Prediction_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print("\n✅ Notebook executed successfully!")
    print("✅ Output saved to: Engine_Energy_Prediction_executed.ipynb")
    
except Exception as e:
    print(f"\n❌ Error during execution: {str(e)}")
    sys.exit(1)
