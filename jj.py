import math
import cmath

# Given data
Vm = 150  # Maximum voltage
R = 40.0  # Resistance
L = 80.0e-3  # Inductance
C = 50.0e-6  # Capacitance


# Angular frequency
omega = 100

# Impedances
Z_R = R
Z_L = 1j * omega * L
Z_C = 1 / (1j * omega * C)

# Total impedance
Z_total = Z_R + Z_L + Z_C

# Current in the circuit
I = Vm / abs(Z_total)

# Voltages across the resistor, inductor, and capacitor
VR0 = I * abs(Z_R)
VL0 = I * abs(Z_L)
VC0 = I * abs(Z_C)

# Represent the voltages as complex numbers
VR0_complex = VR0
VL0_complex = VL0 * cmath.exp(1j * cmath.phase(Z_L))  # The voltage across the inductor leads by 90 degrees
VC0_complex = VC0 * cmath.exp(1j * cmath.phase(Z_C))  # The voltage across the capacitor lags by 90 degrees

# Add the complex voltages
V_total_complex = VR0_complex + VL0_complex + VC0_complex

# The magnitude of the total voltage is the absolute value of the complex total voltage
V_total = abs(V_total_complex)

# Now you can check if V_total is close to Vm
if math.isclose(V_total, Vm, rel_tol=1e-9):
    print("The voltages are approximately equal.")
else:
    print("The voltages are not equal.")