from sympy import symbols, Eq, solve, tan, rad

# Symbols
R = symbols('R')

# Data
omega = 200  # rad/s
L = 0.4  # H
phi = 30  # degrees

# Equation for the resistance
eq = Eq(R, omega * L / tan(rad(phi)))

# Solve the equation for R
solution = solve(eq, R)

# Print the solution
print(f"The resistance R is {solution[0]} ohms.")