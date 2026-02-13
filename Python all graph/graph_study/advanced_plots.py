import matplotlib.pyplot as plt
import numpy as np

def advanced_plots():
    # 1. 3D Scatter Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Data
    x = np.random.standard_normal(100)
    y = np.random.standard_normal(100)
    z = np.random.standard_normal(100)
    c = np.random.standard_normal(100)
    
    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    ax.set_title("3D Scatter Plot")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    fig.colorbar(img)
    plt.show()
    print("Displayed 3D Scatter Plot")

    # 2. 3D Surface Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("3D Surface Plot")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    print("Displayed 3D Surface Plot")

    # 3. Subplots (Multiple plots in one figure)
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle('Subplots Example')
    
    ax1.plot(x, y1, 'r-')
    ax1.set_ylabel('Sin(x)')
    ax1.grid(True)
    
    ax2.plot(x, y2, 'b-')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Cos(x)')
    ax2.grid(True)
    
    plt.show()
    print("Displayed Subplots")

if __name__ == "__main__":
    print("Generating Advanced Plots...")
    advanced_plots()
