from pathlib import Path
import matplotlib.pyplot as plt


#metrisis tou Cs
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Cs.txt").open() as f:
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
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Am.png"))


#metrisis tou Am
mes = dict()
with Path(__file__).with_name("Lab4_measurements_Am.txt").open() as f:
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
plt.savefig(Path(__file__).with_name("Lab4_spectrum_Am.png"))
