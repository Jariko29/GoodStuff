from pathlib import Path
import math as m
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


names = ["spectrum_am_2cm_300s.txt","spectrum_am_4cm_300s.txt","spectrum_am_6cm_300s.txt","spectrum_am_8cm_300s.txt","spectrum_am_10cm_300s.txt","spectrum_am_12cm_300s.txt","spectrum_am_14cm_300s.txt"]
namespng = ["spectrum_am_2cm_300s.png","spectrum_am_4cm_300s.png","spectrum_am_6cm_300s.png","spectrum_am_8cm_300s.png","spectrum_am_10cm_300s.png","spectrum_am_12cm_300s.png","spectrum_am_14cm_300s.png"]
for i in range(len(names)):
    plot_spectrum(names[i],namespng[i])

namescal = ['cal_am_5cm_300s.txt','cal_ra_5cm_800s.txt']
namescalpng = ['cal_am_5cm_300s.png','cal_ra_5cm_800s.png']
for i in range(len(namescal)):
    plot_spectrum(namescal[i],namescalpng[i])



