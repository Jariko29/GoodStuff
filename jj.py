import scipy.integrate as spi
from scipy.misc import derivative
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
    return R * current(t) + L * derivative(current, t, dx=1e-6)

# Integrate to find the energy supplied by the battery
result, _ = spi.quad(lambda t: voltage(t) * current(t), 0, L/R)

# Integrate to find the energy dissipated in the resistor
result_resistor, _ = spi.quad(lambda t: R * current(t)**2, 0, L/R)

print(f"The energy dissipated in the resistor is approximately {result_resistor:.2f} Joules.")

# Integrate to find the energy stored in the inductor
result_inductor, _ = spi.quad(lambda t: L * derivative(current, t, dx=1e-6) * current(t), 0, L/R)

print(f"The energy stored in the inductor is approximately {result_inductor:.2f} Joules.")


print(f"The energy supplied by the battery is approximately {result:.2f} Joules.")