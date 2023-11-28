import matplotlib.pyplot as plt
import numpy as np

# Sample voltage vs. time data
time = np.array([0, 1, 2, 3, 4, 5])  # replace with your time values
voltage = np.array([0, 16, 16, 9, 0, 0])  # replace with your voltage values

# Calculate the derivative of voltage (dV/dt)
dV_dt = np.gradient(voltage, time)

# Constants
epsilon_0 = 8.85e-12  # permittivity of free space
area = 0.001  # area of capacitor plates (replace with your value)

# Calculate displacement current (Id)
displacement_current = epsilon_0 * area * dV_dt

# Plot the graphs
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time, voltage, label='Voltage vs. Time')
plt.title('Voltage vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, displacement_current, label='Displacement Current vs. Time', color='orange')
plt.title('Displacement Current vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Displacement Current (A)')
plt.legend()

plt.tight_layout()
plt.savefig('jj.png')
plt.show()
