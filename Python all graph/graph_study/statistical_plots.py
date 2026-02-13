import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def statistical_plots():
    # Setup data (using built-in datasets)
    tips = sns.load_dataset('tips')
    iris = sns.load_dataset('iris')
    
    # 1. Box Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="day", y="total_bill", data=tips, palette="Set3")
    plt.title("Box Plot: Distribution & Outliers")
    plt.show()
    print("Displayed Box Plot")

    # 2. Violin Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="day", y="total_bill", data=tips, palette="muted", split=True)
    plt.title("Violin Plot: Density & Distribution")
    plt.show()
    print("Displayed Violin Plot")

    # 3. Heatmap
    # Computing correlation matrix
    numeric_iris = iris.select_dtypes(include=[np.number])
    corr = numeric_iris.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Heatmap: Correlation Matrix")
    plt.show()
    print("Displayed Heatmap")

    # 4. Regplot (Regression Plot)
    plt.figure(figsize=(8, 6))
    sns.regplot(x="total_bill", y="tip", data=tips, color='b', line_kws={'color':'red'})
    plt.title("Regression Plot: Linear Relationship")
    plt.show()
    print("Displayed Regression Plot")

    # 5. Pair Plot (Multivariate Analysis)
    print("Generating Pair Plot (this may take a moment)...")
    sns.pairplot(iris, hue="species", palette="husl")
    plt.suptitle("Pair Plot: Multivariate Relationships", y=1.02)
    plt.show()
    print("Displayed Pair Plot")

if __name__ == "__main__":
    print("Generating Statistical Plots...")
    statistical_plots()
