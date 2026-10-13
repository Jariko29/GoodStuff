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


plot_spectrum("spectrum_au_0_120s.txt", "spectrum_au_0_120s.png")
plot_spectrum("spectrum_au_5_120s.txt", "spectrum_au_5_120s.png")
plot_spectrum("spectrum_au_10_360s.txt", "spectrum_au_10_360s.png")
plot_spectrum("spectrum_au_15_600s.txt", "spectrum_au_15_600s.png")
plot_spectrum("spectrum_au_20_1200s.txt", "spectrum_au_20_1200s.png")
plot_spectrum("spectrum_au_25_1900s.txt", "spectrum_au_25_1900s.png")
plot_spectrum("spectrum_au_30_4200s.txt", "spectrum_au_30_4200s.png")