from pathlib import Path
from scipy import constants as cs
import matplotlib.pyplot as plt
import re
import numpy as np

class const:
    AVOG = cs.Avogadro
    '''Avogadro constant'''
    ACTIV = 333000
    '''Activity of source 333000 decays per second'''
    STARG = 60 * 10**-6 
    '''(m^2) Area of target'''
    SDETECT = 14 * 10**-6 
    '''(m^2) Area of detector'''
    R1 = 28 * 10**-3 
    '''(m) distance source -> target'''
    R2 = 26 * 10**-3 
    '''(m) distance target -> detector'''
    XAU = 3.86 * 10**-3 
    '''(g/cm^2) surface density of Au''' 
    AAU = 197 
    '''mazikos arithmos Au'''
    PI = cs.pi

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

def get_count(file_name):
    with Path(__file__).with_name(file_name).open() as f:
        count_sum = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                count_sum += int(parts[1])
    print(f'Sum count of {file_name}: {count_sum}')
    return count_sum
                

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

def effect_cross_sec(count_rate):
    delta_omega_target = const.STARG / const.R1**2
    delta_omega_detector = const.SDETECT / const.R2**2

    N0 = const.ACTIV * delta_omega_target / (4 * const.PI)
    N_target = const.XAU * const.AVOG / const.AAU

    return count_rate / (
        delta_omega_detector * N0 * N_target
    )

# Calibration
mes = read_spectrum("spectrum_Ra_900s.txt")
peak_ranges = [(3900, 4000, 4.78), (5600, 5900, 6.00),  (7700, 7900, 7.68)]
calibration_points = []
for low, high, energy in peak_ranges:
    peak = max(
        (pulse_height for pulse_height in mes if low <= pulse_height <= high),key=mes.get,
    )
    calibration_points.append((peak, energy))

Emax = [pulse_height for pulse_height, _ in calibration_points]
known_energies_MeV = [energy for _, energy in calibration_points]

print("Calibration points (pulse height, energy MeV):", calibration_points)
# Best-fit line: energy = a * pulse_height + b
mean_emax = sum(Emax) / len(Emax)
mean_energy = sum(known_energies_MeV) / len(known_energies_MeV)

a = sum(
    (pulse_height - mean_emax) * (energy - mean_energy)
    for pulse_height, energy in calibration_points
) / sum((pulse_height - mean_emax) ** 2 for pulse_height in Emax)

b = mean_energy - a * mean_emax
print(f"Calibration line: E = {a:.6f} * pulse height {b:+.4f} MeV")

plot_spectrum("spectrum_Ra_900s.txt","spectrum_Ra_900s.png",vertical_lines=Emax,)
plt.figure(figsize=(8, 6))
plt.scatter(Emax, known_energies_MeV, color="#287c78", edgecolor="black", zorder=3)

for pulse_height, energy in calibration_points:
    plt.annotate(
        f"{energy:.2f} MeV",
        (pulse_height, energy),
        xytext=(6, 6),
        textcoords="offset points",
    )

line_x = [min(Emax), max(Emax)]
line_y = [a * pulse_height + b for pulse_height in line_x]
plt.plot(
    line_x,
    line_y,
    color="#c41c1c",
    label=f"E = {a:.6f} * pulse height {b:+.4f} MeV",
)

plt.xlabel("Pulse height")
plt.ylabel("Energy (MeV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("Lab6_calibration.png"), dpi=300)
plt.close()

nds = []
names = ["spectrum_au_0_120s.txt","spectrum_au_5_120s.txt","spectrum_au_10_360s.txt","spectrum_au_15_600s.txt","spectrum_au_20_1200s.txt","spectrum_au_25_1900s.txt","spectrum_au_30_4200s.txt"]
namespng = ["spectrum_au_0_120s.png","spectrum_au_5_120s.png","spectrum_au_10_360s.png","spectrum_au_15_600s.png","spectrum_au_20_1200s.png","spectrum_au_25_1900s.png","spectrum_au_30_4200s.png"]
for i in range(len(names)):
    plot_spectrum(names[i],namespng[i])
    nds.append(effect_cross_sec(get_count(names[i])/int(re.search(r'_(\d+)s\.txt$',names[i]).group(1))))
print(nds)
ndsbarn = [nds[i] *10**24 for i in range(len(nds))]


angles_deg = np.arange(0, 31, 5)
theta_deg = np.linspace(0.1, 50, 1000)  


measured_angles = angles_deg[1:]
measured_values = np.array(ndsbarn[1:])
measured_shape = 1 / np.sin(np.deg2rad(measured_angles / 2))**4
scale = np.dot(measured_shape, measured_values) / np.dot(measured_shape, measured_shape)

curve = scale / np.sin(np.deg2rad(theta_deg / 2))**4

plt.figure(figsize=(8, 6))
plt.plot(theta_deg, curve, label=r"Scaled $1/\sin^4(\theta/2)$")
plt.scatter(angles_deg, ndsbarn, color="tab:red", edgecolor="black",
            zorder=3, label="ndsbarn")
plt.xlabel(r"Angle $\theta$ (degrees)")
plt.ylabel("Cross section (barns)")
plt.xlim(0, 50)
plt.ylim(0, max(ndsbarn) * 1.1)  
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("ndsbarn_vs_rutherford.png", dpi=300)
plt.show()