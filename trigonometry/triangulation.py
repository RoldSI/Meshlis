import numpy as np
from scipy.optimize import fsolve

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Microphone positions
ax, ay = 0, 0
bx, by = 5, 0
cx, cy = 0, 5

# Triangulated point for testing
ox_master, oy_master = 53456, 53456

def equations(vars):
    ox, oy = vars
    eq1 = (distance(ax, ay, ox, oy) - distance(bx, by, ox, oy)) - (distance(ax, ay, ox_master, oy_master) - distance(bx, by, ox_master, oy_master))
    eq2 = (distance(ax, ay, ox, oy) - distance(cx, cy, ox, oy)) - (distance(ax, ay, ox_master, oy_master) - distance(cx, cy, ox_master, oy_master))
    return [eq1, eq2]

# Initial guess for the solution
initial_guess = [0, 0]

# Solve the system of equations
solution = fsolve(equations, initial_guess)
print(f"Ox: {solution[0]}, Oy: {solution[1]}")
