from pathlib import Path
from math import cos,sin, degrees, radians
import matplotlib.pyplot as plt


def read_spectrum(file_name):
    mes = {}
    with Path(__file__).with_name(file_name).open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                key = float(parts[0])
                value = int(parts[1])
                mes[key] = value
    return mes


def plot_spectrum(file_name, output_name):
    mes = read_spectrum(file_name)
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
    plt.savefig(Path(__file__).with_name(output_name), dpi=300)
    plt.close()


#Metrisis tou Cs
plot_spectrum("spectrum_cs_600s.txt", "Lab5_spectrum_Cs.png")

#Metrisis tou Am
plot_spectrum("spectrum_am_600s.txt", "Lab5_spectrum_Am.png")

# Calibration 
Emax = []
for spectrum_name in ["spectrum_cs_600s.txt", "spectrum_am_600s.txt"]:
    mes = read_spectrum(spectrum_name)
    Emax.append(max(mes, key=mes.get))

known_energies_keV = [32.0, 59.5]
source_labels = ["Cs-137", "Am-241"]
print("Emax (Cs, Am):", Emax)

mean_emax = sum(Emax) / len(Emax)
mean_energy = sum(known_energies_keV) / len(known_energies_keV)
a = sum(
    (pulse_height - mean_emax) * (energy - mean_energy)
    for pulse_height, energy in zip(Emax, known_energies_keV)
) / sum((pulse_height - mean_emax) ** 2 for pulse_height in Emax)
b = mean_energy - a * mean_emax
print(f"Calibration line: E = {a:.4f} * Emax {b:.1f} keV")

plt.figure(figsize=(8, 6))
plt.scatter(Emax, known_energies_keV, color="#287c78", edgecolor="black", zorder=3)
for pulse_height, energy, source in zip(Emax, known_energies_keV, source_labels):
    plt.annotate(source, (pulse_height, energy), xytext=(6, 6), textcoords="offset points")

line_x = [min(Emax), max(Emax)]
line_y = [a * pulse_height + b for pulse_height in line_x]
plt.plot(line_x, line_y, color="#c41c1c", label=f"E = {a:.4f} Emax {b:.1f} keV")
plt.xlabel("Pulse height (Emax)")
plt.ylabel("Energy (keV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("Lab5_calibration.png"), dpi=300)
plt.close()

# ola ta spectra
for file_name, output_name in [
    ("spectrum_Ag_1200s.txt", "Lab5_spectrum_Ag.png"),
    ("spectrum_BaCl2_1200s.txt", "Lab5_spectrum_BaCl2.png"),
    ("spectrum_I2_1200s.txt", "Lab5_spectrum_I2.png"),
    ("spectrum_Mo_1200s.txt", "Lab5_spectrum_Mo.png"),
    ("spectrum_SrSO_1200s.txt", "Lab5_spectrum_SrSO.png"),
]:
    plot_spectrum(file_name, output_name)
