import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a sample multivariable function: f(x, y) = sin(x) + cos(y)
def f(x, y):
    return np.sin(x) + np.cos(y)

# Compute the gradient of f at point (a, b)
def gradient_f(a, b):
    grad_x = np.cos(a)
    grad_y = -np.sin(b)
    return np.array([grad_x, grad_y])

# Compute the Hessian of f at point (a, b)
def hessian_f(a, b):
    h_xx = -np.sin(a)
    h_xy = 0
    h_yx = 0
    h_yy = -np.cos(b)
    return np.array([[h_xx, h_xy], [h_yx, h_yy]])

# Taylor approximation up to second order around point (a, b)
# Using einsum for vectorized computation over arrays
def taylor_approx(x, y, a, b):
    delta = np.array([x - a, y - b])
    grad = gradient_f(a, b)
    hess = hessian_f(a, b)
    linear_term = np.einsum('i,i...->...', grad, delta)
    hd = np.einsum('ij,j...->i...', hess, delta)
    quadratic_term = 0.5 * np.einsum('i...,i...->...', delta, hd)
    return f(a, b) + linear_term + quadratic_term

# Numerical comparison at specific points
def numerical_comparison(points, a, b):
    print("Numerical Comparison:")
    print("{:<10} {:<10} {:<15} {:<15} {:<15}".format("x", "y", "Original f", "Taylor Approx", "Error"))
    for x, y in points:
        original = f(x, y)
        approx = taylor_approx(x, y, a, b)
        error = abs(original - approx)
        print("{:<10.4f} {:<10.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(x, y, original, approx, error))

# Graphical representation using 3D surface plots
def graphical_comparison(a, b, x_range=(-2*np.pi, 2*np.pi), y_range=(-2*np.pi, 2*np.pi), grid_size=50):
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    Z_original = f(X, Y)
    Z_approx = taylor_approx(X, Y, a, b)
    
    fig = plt.figure(figsize=(12, 6))
    
    # Original function plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z_original, cmap='viridis')
    ax1.set_title('Original Function f(x, y) = sin(x) + cos(y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x, y)')
    
    # Taylor approximation plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, Z_approx, cmap='viridis')
    ax2.set_title('Second-Order Taylor Approximation around ({:.2f}, {:.2f})'.format(a, b))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Approx')
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Expansion point
    a, b = 0.0, 0.0
    
    # Numerical test points around (a, b)
    test_points = [
        (0.1, 0.1),
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 2.0)  # Farther point to show approximation degradation
    ]
    
    numerical_comparison(test_points, a, b)
    graphical_comparison(a, b)
