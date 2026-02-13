import json

notebook_path = 'digit_recognition.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Final Output Testing (Single Prediction)\n",
                "Visualizing a single prediction from the test set as requested."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# You can change this index to see different digits from the test set\n",
                "index = 2000 \n",
                "\n",
                "# Get the prediction for this specific index\n",
                "# Note: X_test is a DataFrame, so we use .iloc to get the row\n",
                "prediction = clf.predict(X_test.iloc[[index]])[0]\n",
                "\n",
                "print(\"Predicted ======> \" + str(prediction))\n",
                "\n",
                "plt.axis('off')\n",
                "plt.imshow(X_test.iloc[index].values.reshape((28, 28)), cmap='gray')\n",
                "plt.show()"
            ]
        }
    ]

    nb['cells'].extend(new_cells)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Notebook updated successfully with new cells.")

except Exception as e:
    print(f"Error updating notebook: {e}")
