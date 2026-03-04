"""
problem2.py
Finds the intersection points of a circle (y - y1)^2 + (x - x1)^2 = r^2
and a parabola y = a * x^2 + b.
Allows changing the circle center (x1, y1), radius r, parabola width a, and offset b.
Plots the curves and intersection points from -10 to 10 on both axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ============================================
# USER-ADJUSTABLE PARAMETERS
# ============================================
# Circle: (x - x1)^2 + (y - y1)^2 = r^2
x1 = 1.0
y1 = 0.0
r = 4.0          # radius (so r^2 = 16)

# Parabola: y = a * x^2 + b
a = 0.5
b = 1.0

# Plotting range
x_min, x_max = -10, 10
y_min, y_max = -10, 10

# ============================================
# Functions defining the curves
# ============================================
def parabola(x):
    """Return y on the parabola."""
    return a * x**2 + b

def circle_implicit(x, y):
    """Left-hand side of circle equation minus r^2."""
    return (y - y1)**2 + (x - x1)**2 - r**2

def combined_implicit(x):
    """
    Implicit function of x after substituting parabola into circle.
    Returns (parabola(x) - y1)^2 + (x - x1)^2 - r^2.
    Roots of this function correspond to x-coordinates of intersections.
    """
    y = parabola(x)
    return (y - y1)**2 + (x - x1)**2 - r**2

# ============================================
# Find intersection points
# ============================================
# Generate a dense grid to detect sign changes
x_grid = np.linspace(x_min, x_max, 2000)
f_grid = combined_implicit(x_grid)

# List to store intersection points (x, y)
intersections = []

# Check for sign changes (crossings)
for i in range(len(x_grid) - 1):
    f1 = f_grid[i]
    f2 = f_grid[i+1]
    if f1 * f2 < 0:          # sign change -> a root in between
        x0 = (x_grid[i] + x_grid[i+1]) / 2.0
        x_root = fsolve(combined_implicit, x0)[0]
        # Avoid duplicates by checking distance from already found points
        if not any(abs(x_root - xp) < 1e-6 for xp, _ in intersections):
            y_root = parabola(x_root)
            intersections.append((x_root, y_root))
    elif abs(f1) < 1e-8:     # exact zero at grid point (rare)
        x_root = x_grid[i]
        y_root = parabola(x_root)
        if not any(abs(x_root - xp) < 1e-6 for xp, _ in intersections):
            intersections.append((x_root, y_root))

# Also check the last point
if abs(f_grid[-1]) < 1e-8:
    x_root = x_grid[-1]
    y_root = parabola(x_root)
    if not any(abs(x_root - xp) < 1e-6 for xp, _ in intersections):
        intersections.append((x_root, y_root))

# ============================================
# Print results
# ============================================
print(f"Circle: center ({x1}, {y1}), radius {r}")
print(f"Parabola: y = {a} x^2 + {b}")
print(f"Number of intersections found: {len(intersections)}")
for i, (x, y) in enumerate(intersections):
    print(f"  Intersection {i+1}: x = {x:.6f}, y = {y:.6f}")

# ============================================
# Plotting
# ============================================
plt.figure(figsize=(8, 8))

# Plot parabola
x_plot = np.linspace(x_min, x_max, 500)
y_plot = parabola(x_plot)
plt.plot(x_plot, y_plot, 'b-', label='Parabola')

# Plot circle (parametric)
theta = np.linspace(0, 2*np.pi, 500)
x_circle = x1 + r * np.cos(theta)
y_circle = y1 + r * np.sin(theta)
plt.plot(x_circle, y_circle, 'r-', label='Circle')

# Mark intersection points
if intersections:
    xs, ys = zip(*intersections)
    plt.plot(xs, ys, 'go', markersize=8, label='Intersections')

# Axes and limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Intersection of Circle and Parabola')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # keep aspect ratio

plt.show()