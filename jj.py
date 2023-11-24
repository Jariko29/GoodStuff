import numpy as np
import matplotlib.pyplot as plt

# Circuit parameters
C = 1.0  # Capacitance (in Farads)
L = 1.0  # Inductance (in Henrys)
R = 1.0  # Resistance (in Ohms)

# Time range
t = np.linspace(0, 10, 1000)  # Time values from 0 to 10 seconds

# Charge (Q) and current (I) calculations
Q = np.exp(-R * t / (2 * L)) * np.cos(np.sqrt(1 / (L * C) - (R / (2 * L))**2) * t)
I = -np.sqrt(1 / (L * C) - (R / (2 * L))**2) * np.exp(-R * t / (2 * L)) * np.sin(np.sqrt(1 / (L * C) - (R / (2 * L))**2) * t)

# Plotting Q(t) and I(t)
plt.plot(t, Q, label='Q(t)')
plt.plot(t, I, label='I(t)')
plt.xlabel('Time (s)')
plt.ylabel('Charge (C) / Current (A)')
plt.legend()
plt.grid(True)
plt.show()
quit()
