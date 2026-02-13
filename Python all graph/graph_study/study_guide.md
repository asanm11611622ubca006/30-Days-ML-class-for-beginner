# Python Graph Types Study Guide

This guide accompanies the Python scripts in this folder. It explains different types of graphs, when to use them, and which library is best suited for the task.

## 1. Basic Plots (`basic_plots.py`)

**Library:** `Matplotlib` (Standard for plotting)

- **Line Plot**: Shows trends over time or continuous data. (e.g., Stock prices, temperature changes)
- **Bar Chart**: Compares categories. (e.g., Sales by product, population by country)
- **Scatter Plot**: Shows the relationship between two numerical variables. (e.g., Height vs. Weight)
- **Histogram**: Shows the distribution of a single numerical variable. (e.g., Distribution of exam scores)
- **Pie Chart**: Shows parts of a whole (proportions). (e.g., Market share)

## 2. Statistical Plots (`statistical_plots.py`)

**Library:** `Seaborn` (Built on top of Matplotlib, better for statistics and aesthetics)

- **Box Plot**: Summarizes a set of data measured on an interval scale. Shows median, quartiles, and outliers.
- **Violin Plot**: Similar to a box plot but also shows the probability density (thickness) of the data at different values.
- **Heatmap**: Shows data where individual values are represented as colors. Great for correlation matrices.
- **Regplot**: Plots data and a linear regression model fit. Used to see linear relationships.
- **Pair Plot**: Plots pairwise relationships in a dataset. Good for exploring a whole dataset at once.

## 3. Advanced Plots (`advanced_plots.py`)

**Library:** `Matplotlib` (3D toolkits)

- **3D Scatter/Surface**: Used when you have 3 variables (X, Y, Z) to visualize.
- **Subplots**: How to place multiple different graphs in a single figure window.

## 4. Interactive Plots (`interactive_plots.py`)

**Library:** `Plotly` (Great for web-based or interactive exploration)

- Allows zooming, panning, and hovering over data points to see values.
- **Range Slider**: Useful for time-series data to focus on specific time periods.

## 5. Network Graphs (`network_graphs.py`)

**Library:** `NetworkX`

- **Nodes & Edges**: Used to represent connections/relationships.
- **Use cases**: Social networks, transportation routes, process flows.

## How to Study

1. Run each script using `python script_name.py`.
2. Look at the code to see how the data is prepared and how the plot is called.
3. Try changing the data values or the color settings in the code and re-run to see the effect.
