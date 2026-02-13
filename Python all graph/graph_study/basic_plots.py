import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def basic_plots():
    # Setup data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 12]
    
    # 1. Line Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Sin(x)', color='blue', linestyle='--')
    plt.title("Line Plot: Continuous Data")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Displayed Line Plot")

    # 2. Bar Chart
    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color='green')
    plt.title("Bar Chart: Categorical Comparison")ā
    plt.show()
    print("Displayed Bar Chart")

    # 3. Scatter Plot
    x_scatter = np.random.rand(50)
    y_scatter = np.random.rand(50)
    colors = np.random.rand(50)
    sizes = 1000 * np.random.rand(50)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.5, cmap='viridis')
    plt.title("Scatter Plot: Correlation & Distribution")
    plt.colorbar(label='Color Scale')
    plt.show()
    print("Displayed Scatter Plot")

    # 4. Histogram
    data = np.random.randn(1000)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.title("Histogram: Frequency Distribution")
    plt.show()
    print("Displayed Histogram")

    # 5. Pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=140, colors=['gold', 'yellowgreen', 'lightcoral', 'lightskyblue'])
    plt.title("Pie Chart: Proportions")
    plt.show()
    print("Displayed Pie Chart")

if __name__ == "__main__":
    print("Generating Basic Plots...")
    basic_plots()
