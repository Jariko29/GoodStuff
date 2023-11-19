from scipy.misc import derivative

# Function for i(t)
def current(t):
    return ℇ/R * (1 - np.exp(-R/L * t))

# Function for V(t)
def voltage(t):
    return R * current(t) + L * derivative(current, t, dx=1e-6)

# Integrate to find the energy supplied by the battery
result, _ = spi.quad(lambda t: voltage(t) * current(t), 0, L/R)

print(f"The energy supplied by the battery is approximately {result:.2f} Joules.")