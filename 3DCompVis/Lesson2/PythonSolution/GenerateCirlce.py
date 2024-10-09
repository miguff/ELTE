import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 150
num_outliers = 40
radius = 20

# Generate circular distribution points
angles = np.linspace(0, 2 * np.pi, num_points)
x_circle = radius * np.cos(angles) + np.random.normal(0, 2, num_points)
y_circle = radius * np.sin(angles) + np.random.normal(0, 2, num_points)

# Generate random outliers
x_outliers = np.random.uniform(-30, 30, num_outliers)
y_outliers = np.random.uniform(-30, 30, num_outliers)

x = np.append(x_circle,x_outliers)
y = np.append(y_circle,y_outliers)

# Plot the points
plt.figure(figsize=(6, 6))
plt.scatter(x, y, color='blue', label='Circular Points', alpha=0.6)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.title('Circular Distribution with Outliers')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

