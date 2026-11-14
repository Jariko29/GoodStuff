from pathlib import Path
import math as m
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

def read_spectrum(file_name):
    '''Reads the txt file and return a dictonary of [pulseheight , count]'''
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
    '''Plots the graph of pulsheigh vs count'''
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
    plt.gca().xaxis.set_major_locator(MultipleLocator(1000))
    plt.grid(True, alpha=0.3)
    for pulse_height in vertical_lines:
        plt.axvline(pulse_height, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Pulse height")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name(output_name), dpi=400)
    plt.close()

def max_pulse(file_name, dx=[]):
    mes = {}
    temp = 0
    pulse = 0
    with Path(__file__).with_name(file_name).open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                key = float(parts[0])
                value = int(parts[1])
                mes[key] = value
                if value > temp and dx[0] < key < dx[1]:
                    temp = value
                    pulse = key
    return float(pulse)

def get_count(file_name, sel_section=None):
    with Path(__file__).with_name(file_name).open() as f:
        count_sum = 0
        if (sel_section == None):
            for line in f:
                parts = line.strip().split()
                if (len(parts) == 2):
                    count_sum += int(parts[1])
            return count_sum
        elif (sel_section != None):
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    if (float(parts[0]) > sel_section[0] and float(parts[0]) < sel_section[1]):
                        count_sum += int(parts[1])
            return count_sum
        else:
            return 'wtf is going on here'

def compton(name=''):
    if name == 'cs':
        Eg = 662
    if name == 'co':
        Eg = 1318.44
    if name == 'na':
        Eg = 1250    
    e = Eg/511
    T = Eg * (e * (1 - m.cos(m.radians(180))) / (1 + e * (1 - m.cos(m.radians(180)))))
    return float(T)

def radiation(Io,I,x):
    m = (1/x) * np.log(Io/I)
    return float(m)

#-----------------------------------------------------------------------------------
# 2.1 Βαθμονόμηση φάσματος
#-----------------------------------------------------------------------------------
Emax = []
spectrum_name = ["cal_cs_300s.txt", "cal_co_600s.txt"]
mes = read_spectrum(spectrum_name[0])
Emax.append(max_pulse(spectrum_name[0], dx=[4000, 5500]))
mes = read_spectrum(spectrum_name[1])
Emax.append(max_pulse(spectrum_name[1], dx=[8500, 10000]))

known_energies_keV = [662.0, 1333]
source_labels = ["Cs-137", "Co-60"]
print("Emax (Cs, Co):", Emax)

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
plt.savefig(Path(__file__).with_name("Lab9_cal.png"), dpi=300)
plt.close()
plot_spectrum(spectrum_name[0], "cal_cs_300s.png")
plot_spectrum(spectrum_name[1], "cal_co_600s.png")
name_22 = ['22_co_300s.txt','22_cs_300s.txt','22_na_300s.txt']
name_23_co = ['23_co_1thin_300s.txt','23_co_2thin_300s.txt','23_co_1fat_300s.txt','23_co_1fat_1thin_300s.txt','23_co_1fat_2thin_300s.txt','23_co_1fat_3thin_300s.txt']
name_23_cs = ['23_cs_200s.txt','23_cs_1fat_200s.txt','23_cs_2fat_200s.txt','23_cs_3fat_200s.txt','23_cs_4fat_200s.txt']
name_23_na = ['23_na_200s.txt','23_na_1thin_200s.txt','23_na_2thin_200s.txt','23_na_1fat_200s.txt','23_na_1fat_1thin_200s.txt']

#for i in range(len(name_22)):
#    plot_spectrum(name_22[i], name_22[i][:-4]+'.png')

#for i in range(len(name_23_co)):
#    plot_spectrum(name_23_co[i], name_23_co[i][:-4]+'.png')

for i in range(len(name_23_cs)):
    plot_spectrum(name_23_cs[i], name_23_cs[i][:-4]+'.png')

#for i in range(len(name_23_na)):
#     plot_spectrum(name_23_na[i], name_23_na[i][:-4]+'.png')


#-----------------------------------------------------------------------------------
# 2.2 Μέτρηση ενεργειακών φασμάτων ακτίνων−γ
#-----------------------------------------------------------------------------------
measured_gamma = []
temp = max_pulse(name_22[0],[8500,9500])
measured_gamma.append(a*temp+b)
temp = max_pulse(name_22[1],[4000,5500])
measured_gamma.append(a*temp+b)
temp = max_pulse(name_22[2],[8000,9500])
print(temp)
measured_gamma.append(a*temp+b)
print(f'Measured gamma rays Co-60: {measured_gamma[0]} keV, Cs-137: {measured_gamma[1]} keV, Na-22: {measured_gamma[2]} keV')

print(f'{compton('cs')} in keV {(compton('cs') + 13.70) / 0.1480} in pulseheight [Cs-137]')
print(f'{compton('co')} in keV {(compton('co') + 13.70) / 0.1480} in pulseheight [Co-60]')
print(f'{compton('na')} in keV {(compton('na') + 13.70) / 0.1480} in pulseheight [Na-22]')

print((1333 + 13.70)/0.1480)

entasi = []
entasi.append(get_count(name_22[0], sel_section=[8500,9600]))
entasi.append(get_count(name_22[1], sel_section=[4000,5000]))
entasi.append(get_count(name_22[2], sel_section=[8000,9100]))
entasi = [i/300 for i in entasi]
print(entasi)

entasi_co = []
aporofisi = []
paxos_co = [1.1508,2.2912,5.5769,6.7277,7.8682,9.0171]
for name in name_23_co:
    temp = get_count(name,[8500,9650])
    entasi_co.append(temp/300)
print(f'Intensity tou Co-60 {entasi_co}')
for i in range(len(paxos_co)):
    aporofisi.append(radiation(entasi[0],entasi_co[i],paxos_co[i]))
print(f'Sintelestis aporofisis gia Co-60: {np.average(aporofisi)} gia 7554 keV')



print('\nfin\n')