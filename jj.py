import math

# Data
distance = 5.0  # Distance the astronaut has moved away from the station in meters
power_laser = 1000  # Power of the laser in watts
oxygen_remaining = 10  # Remaining oxygen in hours
mass_astronaut = 70  # Mass of the astronaut in kg

# Constants
speed_of_light = 3.0e8  # Speed of light in meters/second

# Calculations
force = power_laser / speed_of_light
acceleration = force / mass_astronaut
time_to_return = math.sqrt(2 * distance / acceleration)

# Check if the time exceeds the remaining oxygen
if time_to_return > oxygen_remaining * 3600:  # convert oxygen_remaining to seconds
    print("The return is not possible. The time exceeds the remaining oxygen.")
else:
    print(f"The astronaut will need approximately {time_to_return:.2f} seconds to return.")