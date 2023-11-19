import scipy.integrate as spi
import numpy as np

# Given values
ℇ = 12.0  # V
R = 3.0   # Ω
L = 0.600 # H

# Function for i(t)
def current(t):
    return ℇ/R * (1 - np.exp(-R/L * t))

# Function for V(t)
def voltage(t):
    return R * current(t) + L * np.gradient(current(t), t)

# Integrate to find the energy supplied by the battery
result, _ = spi.quad(lambda t: voltage(t) * current(t), 0, L/R)

print(f"The energy supplied by the battery is approximately {result:.2f} Joules.")
