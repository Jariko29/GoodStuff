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


def calculate_fwhm(spectrum):
    points = sorted(spectrum.items())
    if not points:
        raise ValueError("Cannot calculate FWHM for an empty spectrum")

    peak_index = max(range(len(points)), key=lambda index: points[index][1])
    peak_count = points[peak_index][1]
    if peak_count <= 0:
        raise ValueError("Cannot calculate FWHM when the peak count is zero")

    half_max = peak_count / 2

    left_index = peak_index - 1
    while left_index >= 0 and points[left_index][1] >= half_max:
        left_index -= 1
    if left_index < 0:
        raise ValueError("Peak does not cross half maximum on the left")

    x1, y1 = points[left_index]
    x2, y2 = points[left_index + 1]
    left_crossing = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    right_index = peak_index + 1
    while right_index < len(points) and points[right_index][1] >= half_max:
        right_index += 1
    if right_index == len(points):
        raise ValueError("Peak does not cross half maximum on the right")

    x1, y1 = points[right_index - 1]
    x2, y2 = points[right_index]
    right_crossing = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    return right_crossing - left_crossing


def plot_spectrum(file_name, output_name, vertical_lines=()):
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
    for pulse_height in vertical_lines:
        plt.axvline(pulse_height, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Pulse height")
    plt.ylabel("Count")
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
print(f"Calibration line: E = {a:.4f} * Emax {b:+.2f} keV")

plt.figure(figsize=(8, 6))
plt.scatter(Emax, known_energies_keV, color="#287c78", edgecolor="black", zorder=3)
for pulse_height, energy, source in zip(Emax, known_energies_keV, source_labels):
    plt.annotate(source, (pulse_height, energy), xytext=(6, 6), textcoords="offset points")

line_x = [min(Emax), max(Emax)]
line_y = [a * pulse_height + b for pulse_height in line_x]
plt.plot(line_x, line_y, color="#c41c1c", label=f"E = {a:.4f} Emax {b:+.2f} keV")
plt.xlabel("Pulse height (Emax)")
plt.ylabel("Energy (keV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("Lab5_calibration.png"), dpi=300)
plt.close()

sample_spectra = [
    ("Ag", "spectrum_Ag_1200s.txt", "Lab5_spectrum_Ag.png"),
    ("BaCl2", "spectrum_BaCl2_1200s.txt", "Lab5_spectrum_BaCl2.png"),
    ("I2", "spectrum_I2_1200s.txt", "Lab5_spectrum_I2.png"),
    ("Mo", "spectrum_Mo_1200s.txt", "Lab5_spectrum_Mo.png"),
    ("SrSO", "spectrum_SrSO_1200s.txt", "Lab5_spectrum_SrSO.png"),
]
for sample, file_name, output_name in sample_spectra:
    vertical_lines = (720, 1100) if sample == "SrSO" else ()
    plot_spectrum(file_name, output_name, vertical_lines=vertical_lines)

# Plot gia tis energeies twn stoixwn
sample_peak_energies = []
sample_fwhm_energies = []
for sample, file_name, _ in sample_spectra:
    mes = read_spectrum(file_name)
    peak_pulse_height = max(mes, key=mes.get)
    peak_energy = a * peak_pulse_height + b
    peak_count = mes[peak_pulse_height]
    fwhm_pulse_height = calculate_fwhm(mes)
    fwhm_energy = abs(a) * fwhm_pulse_height
    sample_peak_energies.append(peak_energy)
    sample_fwhm_energies.append(fwhm_energy)
    print(
        f"{sample} peak: pulse height = {peak_pulse_height:.2f}, "
        f"count = {peak_count}, energy = {peak_energy:.2f} keV, "
        f"FWHM = {fwhm_pulse_height:.2f} pulse-height units "
        f"({fwhm_energy:.2f} keV)"
    )

print("FWHM values in keV:", [f"{x:.2f}" for x in sample_fwhm_energies])

atomic_numbers = [47, 56, 53, 42, 38]
rydberg_energy_eV = 13.6
theoretical_energies = [
    rydberg_energy_eV * (atomic_number - 1) ** 2 * (1 - 1 / 2**2) / 1000
    for atomic_number in atomic_numbers
]
print("Theoretical K-alpha energies (keV):")
for (sample, _, _), energy in zip(sample_spectra, theoretical_energies):
    print(f"{sample}: {energy:.2f} keV")

sample_indices = range(len(sample_spectra))
plt.figure(figsize=(8, 6))
plt.errorbar(
    sample_indices,
    sample_peak_energies,
    yerr=[width / 2 for width in sample_fwhm_energies],
    fmt="o",
    color="#287c78",
    ecolor="#287c78",
    markeredgecolor="black",
    capsize=4,
    label="Experimental (±FWHM/2)",
)
plt.scatter(
    sample_indices,
    theoretical_energies,
    color="#c41c1c",
    marker="s",
    edgecolor="black",
    s=35,
    label="Theoretical",
)
for index, energy in enumerate(sample_peak_energies):
    plt.annotate(
        f"{energy:.2f}",
        (index, energy),
        xytext=(-7, 8),
        textcoords="offset points",
        ha="right",
        color="#1d625f",
    )
for index, energy in enumerate(theoretical_energies):
    plt.annotate(
        f"{energy:.2f}",
        (index, energy),
        xytext=(7, -11),
        textcoords="offset points",
        color="#a31818",
    )
plt.xticks(sample_indices, [sample for sample, _, _ in sample_spectra])
plt.xlabel("Target")
plt.ylabel(r"$E_{K\alpha}$ (keV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("Lab5_calibrated_spectra_peaks.png"), dpi=300)
plt.close()


# Moseley's law fit
mosley_x = [(atomic_number - 1) ** 2 for atomic_number in atomic_numbers]
mosley_data = [
    (x_value, energy, fwhm, sample)
    for x_value, energy, fwhm, (sample, _, _) in zip(
        mosley_x, sample_peak_energies, sample_fwhm_energies, sample_spectra
    )
    if sample != "SrSO"
]
mosley_x, mosley_energies, mosley_fwhm, mosley_samples = zip(*mosley_data)
mosley_slope = sum(
    x_value * energy for x_value, energy in zip(mosley_x, mosley_energies)
) / sum(x_value**2 for x_value in mosley_x)
mosley_residual_sum_squares = sum(
    (energy - mosley_slope * x_value) ** 2
    for x_value, energy in zip(mosley_x, mosley_energies)
)
mosley_slope_error = (
    mosley_residual_sum_squares
    / ((len(mosley_x) - 1) * sum(x_value**2 for x_value in mosley_x))
) ** 0.5
experimental_rydberg_keV = mosley_slope / (1 - 1 / 2**2)
experimental_rydberg_eV = experimental_rydberg_keV * 1000
experimental_rydberg_error_eV = mosley_slope_error / (1 - 1 / 2**2) * 1000
print(
    f"Moseley fit: E_Kα = ({mosley_slope:.6f} ± "
    f"{mosley_slope_error:.6f}) (Z - 1)^2 keV"
)
print(
    f"Experimental Rydberg energy: {experimental_rydberg_eV:.2f} ± "
    f"{experimental_rydberg_error_eV:.2f} eV"
)

line_x = [min(mosley_x), max(mosley_x)]
line_y = [mosley_slope * x_value for x_value in line_x]
plt.figure(figsize=(8, 6))
plt.errorbar(
    mosley_x,
    mosley_energies,
    yerr=[width / 2 for width in mosley_fwhm],
    fmt="o",
    color="#287c78",
    ecolor="#287c78",
    markeredgecolor="black",
    capsize=4,
)
plt.plot(
    line_x,
    line_y,
    color="#c41c1c",
    label=(
        rf"$E_{{K\alpha}} = {mosley_slope:.5f}(Z-1)^2$ keV"
        "\n"
        rf"$R_{{\infty}}$ = {experimental_rydberg_eV:.2f} eV"
    ),
)
for x_value, peak_energy, sample in zip(mosley_x, mosley_energies, mosley_samples):
    plt.annotate(
        sample,
        (x_value, peak_energy),
        xytext=(7, 5),
        textcoords="offset points",
    )
plt.xlabel(r"$(Z-1)^2$")
plt.ylabel(r"$E_{K\alpha}$ (keV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("Lab5_moseley_law.png"), dpi=300)
plt.close()
