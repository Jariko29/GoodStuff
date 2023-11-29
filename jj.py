from sympy import symbols, Eq, solve, cos, sin, rad

# Symbols
R, Im, phi = symbols('R Im phi')

# Data
omega = 200  # rad/s
Vin_amplitude = 20
Vin_phase = 30  # degrees
L = 0.4  # H

# Equation for the term containing the resistance
eq1 = Eq(Im * R, Vin_amplitude * cos(rad(Vin_phase)))

# Equation for the term containing the inductor term
eq2 = Eq(Im * L * omega, Vin_amplitude * sin(rad(Vin_phase)))

# Solve the system of equations
solution = solve((eq1, eq2), (R, Im))

# Print the solution
print(solution)