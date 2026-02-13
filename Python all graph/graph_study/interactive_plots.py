import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def interactive_plots():
    # Load data
    df = px.data.iris()

    # 1. Interactive Scatter Plot
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                     size='petal_length', hover_data=['petal_width'],
                     title="Interactive Scatter Plot (Plotly)")
    fig.show()
    print("Displayed Interactive Scatter Plot")

    # 2. Interactive 3D Scatter Plot
    fig3d = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                          color='species', symbol='species',
                          title="Interactive 3D Scatter Plot")
    fig3d.show()
    print("Displayed Interactive 3D Scatter Plot")

    # 3. Interactive Line Plot with Range Slider
    # Creating time series data
    dates = pd.date_range(start='2023-01-01', periods=100)
    values = np.random.randn(100).cumsum()
    ts_df = pd.DataFrame({'Date': dates, 'Value': values})
    
    fig_line = px.line(ts_df, x='Date', y='Value', title='Time Series with Range Slider')
    fig_line.update_xaxes(rangeslider_visible=True)
    fig_line.show()
    print("Displayed Interactive Line Plot")

if __name__ == "__main__":
    print("Generating Interactive Plots...")
    interactive_plots()
