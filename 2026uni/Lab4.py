from pathlib import Path
from math import cos, radians
import matplotlib.pyplot as plt


#metrisis tou Cs
Emax = []
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
Emax.append(max((pulse_height for pulse_height in mes if pulse_height > 5000), key=mes.get))
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs.png"))


#metrisis tou Am
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Am.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
Emax.append(max(mes, key=mes.get))
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Am.png"))


#metrisis tou Na
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Na.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
Emax.append(max(mes, key=mes.get))
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Na.png"))

print("Emax (Cs, Am, Na):", Emax)


# Calibration
known_energies_keV = [661.7, 59.5, 511.0]
mean_emax = sum(Emax) / len(Emax)
mean_energy = sum(known_energies_keV) / len(known_energies_keV)
a = sum(
    (pulse_height - mean_emax) * (energy - mean_energy)
    for pulse_height, energy in zip(Emax, known_energies_keV)
) / sum((pulse_height - mean_emax) ** 2 for pulse_height in Emax)
b = mean_energy - a * mean_emax
print(f"Calibration line: E = {a:.6f} * Emax + {b:.6f} keV")

source_labels = ["Cs-137", "Am-241", "Na-22"]
plt.figure(figsize=(8, 6))
plt.scatter(Emax, known_energies_keV, color="#287c78", edgecolor="black", zorder=3)
for pulse_height, energy, source in zip(Emax, known_energies_keV, source_labels):
    plt.annotate(source, (pulse_height, energy), xytext=(6, 6), textcoords="offset points")

line_x = [min(Emax), max(Emax)]
line_y = [a * pulse_height + b for pulse_height in line_x]
plt.plot(line_x, line_y, color="#c41c1c", label=f"E = {a:.6f} Emax  {b:.6f} keV")
plt.xlabel("Pulse height (Emax)")
plt.ylabel("Energy (keV)")
plt.title("Energy calibration")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("Lab4_calibration.png"), dpi=300)
plt.show()


#metrisis tou Cs 10deg 500s
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs10deg_500s.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs10deg_500s.png"))


#metrisis tou Cs 20deg 600s
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs20deg_600s.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs20deg_600s.png"))


#metrisis tou Cs 30deg 600s
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs30deg_600s.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs30deg_600s.png"))


#metrisis tou Cs 40deg 600s
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs40deg_600s.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs40deg_600s.png"))


#metrisis tou Cs 60deg 600s
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs60deg_600s.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs60deg_600s.png"))


#metrisis tou Cs 90deg 600s
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs90deg_600s.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs90deg_600s.png"))


#metrisis tou Cs 120deg 600s
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs120deg_600s.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs120deg_600s.png"))


#metrisis tou Cs 130deg 600s
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs130deg_600s.txt").open() as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            key = float(parts[0])
            value = int(parts[1])
            mes[key] = value
pulse_heights, counts = zip(*sorted(mes.items()))
bin_width = pulse_heights[1] - pulse_heights[0]

plt.figure(figsize=(8, 6))
plt.bar(
    pulse_heights,
    counts,
    width=bin_width,
    align="center",
    edgecolor="black",
    linewidth=0.6,
)
plt.xlabel("Pulse height")
plt.ylabel("Count")
plt.title("Pulse-height spectrum")
plt.tight_layout()
plt.show()
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Cs130deg_600s.png"))


# Right-side peak pulse heights for the Cs angle measurements
angle_measurement_files = [
    "Lab4_measurements_Cs10deg_500s.txt",
    "Lab4_measurements_Cs20deg_600s.txt",
    "Lab4_measurements_Cs30deg_600s.txt",
    "Lab4_measurements_Cs40deg_600s.txt",
    "Lab4_measurements_Cs60deg_600s.txt",
    "Lab4_measurements_Cs90deg_600s.txt",
    "Lab4_measurements_Cs120deg_600s.txt",
    "Lab4_measurements_Cs130deg_600s.txt",
]
E = []
for measurement_file in angle_measurement_files:
    mes = {}
    with Path(__file__).with_name(measurement_file).open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mes[float(parts[0])] = int(parts[1])
    min_pulse_height = 6000 if measurement_file in angle_measurement_files[:3] else 2000
    E.append(max((pulse_height for pulse_height in mes if pulse_height > min_pulse_height), key=mes.get))

print("E (Cs right-side peak pulse heights):", E)



angles = [10, 20, 30, 40, 60, 90, 120, 130]
E_gamma = 661.7
me_c_squared_keV = 511.0
epsilon = E_gamma / me_c_squared_keV
E_prime_gamma = [
    E_gamma / (1 + epsilon * (1 - cos(radians(angle))))
    for angle in angles
]
measured_peak_energies_keV = [a * pulse_height + b for pulse_height in E]
print("Predicted E'gamma (keV):", E_prime_gamma)

plt.figure(figsize=(8, 6))
plt.scatter(
    angles,
    measured_peak_energies_keV,
    color="#287c78",
    edgecolor="black",
    label="Measured peak (calibrated)",
)
plt.scatter(
    angles,
    E_prime_gamma,
    color="#c41c1c",
    marker="s",
    edgecolor="black",
    label="Compton prediction",
)
plt.xlabel("Scattering angle (degrees)")
plt.ylabel("Photon energy (keV)")
plt.title("Compton-scattered photon energy by angle")
plt.xticks(angles)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("Lab4_peak_pulseheight_over_angle.png"), dpi=300)
plt.show()


