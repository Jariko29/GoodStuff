from sympy import symbols, Eq, solve, cos, sin, rad

# Σύμβολα
R, Im, phi = symbols('R Im phi')

# Δεδομένα
omega = 200  # rad/s
Vin_amplitude = 20
Vin_phase = 30  # degrees
L = 0.4  # H

# Εξίσωση για τον όρο που περιέχει την αντίσταση
eq1 = Eq(Im * R, Vin_amplitude * cos(Vin_phase * rad))

# Εξίσωση για τον όρο που περιέχει τον όρο του πηνίου
eq2 = Eq(Im * L * omega, Vin_amplitude * sin(Vin_phase * rad))

# Λύση του συστήματος εξισώσεων
solution = solve((eq1, eq2), (R, Im, phi))

# Εκτύπωση της απάντησης
print(f"The value of R is: {solution[R]:.2f} ohms")
