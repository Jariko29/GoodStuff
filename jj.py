from sympy import symbols, Eq, solve, tan, rad

# Symbols
R = symbols('R')

# Data
omega = 200  # rad/s
L = 0.4  # H
phi1 = 30  # degrees
phi2 = -30  # degrees

# Equation for the resistance
eq1 = Eq(R, omega * L / tan(rad(phi1)))
eq2 = Eq(R, omega * L / tan(rad(phi2)))

# Solve the equation for R
solution1 = solve(eq1, R)
solution2 = solve(eq2, R)

# Print the solutions
print(f"The resistance R for a phase shift of 30 degrees is {solution1[0]} ohms.")
print(f"The resistance R for a phase shift of -30 degrees is {solution2[0]} ohms.")