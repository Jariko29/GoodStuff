import math

def calculate_new_beam_diameter(initial_diameter, wavelength, distance):
    # Convert initial diameter to meters
    initial_diameter_meters = initial_diameter / 1000.0  # converting mm to meters

    # Calculate diffraction-limited beam diameter
    diffraction_limited_diameter = (4 * wavelength * distance) / (math.pi * initial_diameter_meters)

    # Calculate angular spread
    theta = diffraction_limited_diameter / distance

    # Calculate new beam diameter at the given distance
    new_beam_diameter = 2 * distance * math.tan(theta / 2)

    return new_beam_diameter

# Example usage:
initial_beam_diameter = 0.5  # in mm
wavelength = 0.633 * 1e-6  # wavelength for a helium-neon laser in meters
distance = 400000.0  # in meters

new_beam_diameter = calculate_new_beam_diameter(initial_beam_diameter, wavelength, distance)
print(f"New beam diameter at {distance} meters: {new_beam_diameter} meters")
