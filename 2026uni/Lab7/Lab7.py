from pathlib import Path
import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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


def max_pulse(file_name):
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
                if value > temp:
                    temp = value
                    pulse = key
    return float(pulse)


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
    for pulse_height in vertical_lines:
        plt.axvline(pulse_height, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Pulse height")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name(output_name), dpi=400)
    plt.close()


# ---------------------------------------------------------------------------
# Count rate vs distance
# ---------------------------------------------------------------------------
R = []
D = [i for i in range(2, 15, 2)]  # Distance cm
xronos = [300, 350, 400, 400, 450, 450, 500, 550]
print(D)
names = ["spectrum_am_2cm_300s.txt", "spectrum_am_4cm_300s.txt", "spectrum_am_6cm_300s.txt",
         "spectrum_am_8cm_300s.txt", "spectrum_am_10cm_300s.txt", "spectrum_am_12cm_300s.txt",
         "spectrum_am_14cm_300s.txt"]
namespng = ["spectrum_am_2cm_300s.png", "spectrum_am_4cm_300s.png", "spectrum_am_6cm_300s.png",
            "spectrum_am_8cm_300s.png", "spectrum_am_10cm_300s.png", "spectrum_am_12cm_300s.png",
            "spectrum_am_14cm_300s.png"]

N_counts = []  
for i in range(len(names)):
    plot_spectrum(names[i], namespng[i])
    n = get_count(names[i], [900, 5000])
    N_counts.append(n)
    R.append(n / xronos[i])
    print(n / xronos[i])

namescal = ['cal_am_5cm_300s.txt', 'cal_ra_5cm_800s.txt']
namescalpng = ['cal_am_5cm_300s.png', 'cal_ra_5cm_800s.png']
for i in range(len(namescal)):
    plot_spectrum(namescal[i], namescalpng[i])

D = np.array(D, dtype=float)
R = np.array(R, dtype=float)
N = np.array(N_counts, dtype=float)
t = np.array(xronos[:len(names)], dtype=float)

# ---------------------------------------------------------------------------
# Slope of log10(R) vs log10(D) with errors
# ---------------------------------------------------------------------------
sigD = np.full_like(D, 0.5)   # cm, distance uncertainty
sigR = np.sqrt(N) / t         # counts/s, Poisson uncertainty

EXCLUDE_D = [2.0]   # distances (cm) left out of the slope fit; use [] to keep all points
mask = (D > 0) & (R > 0) & ~np.isin(D, EXCLUDE_D)
D, R, N, t, sigD, sigR = D[mask], R[mask], N[mask], t[mask], sigD[mask], sigR[mask]

x = np.log10(D)
y = np.log10(R)
sx = sigD / (D * np.log(10))   # sigma of log10(D)
sy = sigR / (R * np.log(10))   # sigma of log10(R) = 1/(sqrt(N) ln10)

# --- (a) plain fit, scatter only (no input errors) -------------------------
res = stats.linregress(x, y)
print("\n--- (a) Unweighted linregress (scatter only) ---")
print(f"slope = {res.slope:.3f} ± {res.stderr:.3f}")
print(f"R² = {res.rvalue**2:.3f}")

# --- (b) weighted polyfit with effective variance --------------------------
# sigma_i^2 = sy_i^2 + m^2 * sx_i^2   (iterate because m is unknown)
m_fit = np.polyfit(x, y, 1)[0]
for _ in range(50):
    s_eff = np.sqrt(sy**2 + (m_fit * sx) ** 2)
    coef, cov = np.polyfit(x, y, 1, w=1 / s_eff, cov='unscaled')
    if abs(coef[0] - m_fit) < 1e-10:
        m_fit = coef[0]
        break
    m_fit = coef[0]

slope, intercept = coef
slope_err = np.sqrt(cov[0, 0])                      # from input errors only
chi2 = np.sum(((y - np.polyval(coef, x)) / s_eff) ** 2)
red_chi2 = chi2 / (len(x) - 2)
slope_err_scaled = slope_err * np.sqrt(max(red_chi2, 1.0))

print("\n--- (b) Weighted polyfit, effective variance ---")
print(f"slope = {slope:.3f} ± {slope_err:.3f}   (input errors only)")
print(f"slope = {slope:.3f} ± {slope_err_scaled:.3f}   (scaled by sqrt(chi2_red) if > 1)")
print(f"chi2 = {chi2:.2f}, reduced chi2 = {red_chi2:.2f}")
print(f"deviation from -2: {abs(slope + 2) / slope_err_scaled:.1f} sigma")

# --- (c) Monte Carlo with polyfit ------------------------------------------
rng = np.random.default_rng(0)
slopes = []
for _ in range(10000):
    Dt = rng.normal(D, sigD)    # distance error 0.5 cm
    Rt = rng.normal(R, sigR)    # counting error
    ok = (Dt > 0) & (Rt > 0)
    slopes.append(np.polyfit(np.log10(Dt[ok]), np.log10(Rt[ok]), 1)[0])
slopes = np.array(slopes)

slope0 = np.polyfit(x, y, 1)[0]
print("\n--- (c) Monte Carlo (10000 trials) ---")
print(f"slope = {slope0:.3f} ± {np.std(slopes, ddof=1):.3f}")

# --- plot with error bars --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.errorbar(D, R, xerr=sigD, yerr=sigR, fmt='o', color='red',
             ecolor='black', capsize=3, markeredgecolor='black')
Ds = np.linspace(D.min(), D.max(), 100)
plt.plot(Ds, 10**intercept * Ds**slope, 'r-',
         label=f"slope = {slope:.2f} ± {slope_err_scaled:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.grid(True, alpha=0.3, which='both')
plt.xlabel('Distance (cm)')
plt.ylabel('Count rate (counts/s)')
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name('countsec_vs_distance.png'), dpi=300)
plt.show()

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
mes = read_spectrum("cal_ra_5cm_800s.txt")
peak_ranges = [(3900, 4000, 4.78), (5600, 5900, 6.00), (7700, 7900, 7.68)]
calibration_points = []
for low, high, energy in peak_ranges:
    peak = max((pulse_height for pulse_height in mes if low <= pulse_height <= high), key=mes.get)
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

# --- Error on the calibration slope from the FWHM of the three peaks -------
# sigma_P = FWHM / 2.355 is the uncertainty of each peak position (pulse height).
# The known energies are treated as exact, so the pulse-height error is turned
# into an energy error  sigma_E,i = a * sigma_P,i  (effective variance).
FWHM_TO_SIGMA_CAL = 1 / (2 * np.sqrt(2 * np.log(2)))
def peak_fwhm(spectrum, peak, search=300, smooth=3):
    '''FWHM of the peak located near `peak` (pulse height).
    Works inside +-search channels, smooths the counts with a moving average to
    suppress noise and measures the half height above a baseline, so the peak does
    not need to be the global maximum or isolated. Returns nan if the peak does not
    fall to half height on both sides inside the window.'''
    pts = sorted((k, v) for k, v in spectrum.items() if abs(k - peak) <= search)
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    ys = np.convolve(y, np.ones(smooth) / smooth, mode='same')
    i0 = int(np.argmin(np.abs(x - peak)))
    lo, hi = max(i0 - 5 * smooth, 0), min(i0 + 5 * smooth + 1, len(x))
    ip = lo + int(np.argmax(ys[lo:hi]))          # local maximum next to the given peak
    base = ys.min()
    half = base + (ys[ip] - base) / 2
    il = ip
    while il > 0 and ys[il] >= half:
        il -= 1
    ir = ip
    while ir < len(x) - 1 and ys[ir] >= half:
        ir += 1
    if ys[il] >= half or ys[ir] >= half:
        return np.nan
    xl = x[il] + (half - ys[il]) * (x[il + 1] - x[il]) / (ys[il + 1] - ys[il])
    xr = x[ir - 1] + (half - ys[ir - 1]) * (x[ir] - x[ir - 1]) / (ys[ir] - ys[ir - 1])
    return xr - xl


def measure_fwhm(spectrum, peak):
    '''Tries wider and wider windows until the peak drops below half height on both sides.'''
    for search in (300, 600, 1000):
        fw = peak_fwhm(spectrum, peak, search=search)
        if not np.isnan(fw):
            return fw
    raise ValueError(f"Could not measure the FWHM of the peak at {peak:.0f}; "
                     f"look at the spectrum around it (plot it) and adjust the windows")


FWHM_CAL = [400, 340, 320]   # channels, for the 4.78, 6.00 and 7.68 MeV peaks; set to None to measure them

sigP_cal = []
for idx, pk in enumerate(Emax):
    fw = FWHM_CAL[idx] if FWHM_CAL is not None else measure_fwhm(mes, pk)
    sigP_cal.append(fw * FWHM_TO_SIGMA_CAL)
    print(f"Peak at {pk:.0f}: FWHM = {fw:.1f} channels, sigma_P = {fw * FWHM_TO_SIGMA_CAL:.1f}")

Pc_cal = np.array(Emax, dtype=float)
Ec_cal = np.array(known_energies_MeV, dtype=float)
sP_cal = np.array(sigP_cal, dtype=float)

a_w = a   # start from the unweighted slope and iterate
for _ in range(100):
    sE_cal = a_w * sP_cal
    w_cal = 1 / sE_cal ** 2
    S_ = w_cal.sum()
    Sx_ = (w_cal * Pc_cal).sum()
    Sy_ = (w_cal * Ec_cal).sum()
    Sxx_ = (w_cal * Pc_cal ** 2).sum()
    Sxy_ = (w_cal * Pc_cal * Ec_cal).sum()
    Dl_ = S_ * Sxx_ - Sx_ ** 2
    a_new = (S_ * Sxy_ - Sx_ * Sy_) / Dl_
    b_w = (Sxx_ * Sy_ - Sx_ * Sxy_) / Dl_
    if abs(a_new - a_w) < 1e-14:
        a_w = a_new
        break
    a_w = a_new

sig_a_cal = np.sqrt(S_ / Dl_)       # error on the slope from the FWHM only
sig_b_cal = np.sqrt(Sxx_ / Dl_)
chi2_cal = np.sum(((Ec_cal - (a_w * Pc_cal + b_w)) / sE_cal) ** 2)
red_chi2_cal = chi2_cal / (len(Pc_cal) - 2)       # only 1 degree of freedom with 3 points
sig_a_scaled = sig_a_cal * np.sqrt(max(red_chi2_cal, 1.0))

# Monte Carlo check: shift each peak by its sigma_P and refit with polyfit
rng_cal = np.random.default_rng(1)
a_mc = [np.polyfit(rng_cal.normal(Pc_cal, sP_cal), Ec_cal, 1)[0] for _ in range(10000)]

# Scatter-only error (unweighted, residuals around the line) for comparison
s2_cal = np.sum((Ec_cal - (a * Pc_cal + b)) ** 2) / (len(Pc_cal) - 2)
sig_a_scatter = np.sqrt(s2_cal / np.sum((Pc_cal - Pc_cal.mean()) ** 2))

print("\n--- Error on the calibration slope ---")
print(f"weighted slope a = {a_w:.6f} ± {sig_a_cal:.6f} MeV/channel (FWHM only)")
print(f"weighted slope a = {a_w:.6f} ± {sig_a_scaled:.6f} MeV/channel (scaled by sqrt(chi2_red) if > 1)")
print(f"b = {b_w:.4f} ± {sig_b_cal:.4f} MeV, chi2_red = {red_chi2_cal:.2f}")
print(f"Monte Carlo (polyfit): a = {np.polyfit(Pc_cal, Ec_cal, 1)[0]:.6f} ± {np.std(a_mc, ddof=1):.6f}")
print(f"Scatter only (residuals): a = {a:.6f} ± {sig_a_scatter:.6f}")

plot_spectrum("cal_ra_5cm_800s.txt", "cal_ra_5cm_800s.png", vertical_lines=Emax,)
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
plt.savefig(Path(__file__).with_name("Lab7_calibration.png"), dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Energy loss in obstacles
# ---------------------------------------------------------------------------
namepen = ['spectrum_am_5cm_alum_300s.txt', 'spectrum_am_5cm_mylar_300s.txt']
namepen_png = ['spectrum_am_5cm_alum_300s.png', 'spectrum_am_5cm_mylar_300s.png']
for i in range(len(namepen)):
    plot_spectrum(namepen[i], namepen_png[i])

int_E = 5.48  # MeV
E_at_target = 5  # MeV
det_E_no_obs = np.round(0.000772 * max_pulse('cal_am_5cm_300s.txt') + 1.7152, 3)
print(f'Energy on the detectror with no  obs: {det_E_no_obs} MeV')
det_E_with_mylar = np.round(0.000772 * max_pulse('spectrum_am_5cm_mylar_300s.txt') + 1.7152, 3)
print(f'\nEnergy with mylar obst: {det_E_with_mylar} MeV')
E_alpha_mylar = E_at_target - ((E_at_target - det_E_with_mylar) / 2)
print(f'Energy of alpha particle: {E_alpha_mylar} MeV')
dedx_mylar = 0.127  # MeV/μm
dx_mylar = (E_at_target - det_E_with_mylar) / (dedx_mylar)
print(f'Width of mylar: {dx_mylar} μm')

det_E_with_alum = np.round(0.000772 * max_pulse('spectrum_am_5cm_alum_300s.txt') + 1.7152, 3)
print(f'\nEnergy with aluminium obs: {det_E_with_alum} MeV')
E_alpha_alum = E_at_target - ((E_at_target - det_E_with_alum) / 2)
print(f'Energy of alpha particle (alum): {E_alpha_alum} MeV')
dedx_alum = 0.138  # MeV/μm
dx_alum = (E_at_target - det_E_with_alum) / dedx_alum
print(f'Width of alum: {dx_alum} μm')

Energy = [2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6]
Aluminium = [0.257, 0.246, 0.237, 0.228, 0.220, 0.212, 0.205, 0.198, 0.192, 0.187, 0.181, 0.176, 0.172, 0.167, 0.163, 0.159, 0.155]
Mylar = [0.196, 0.186, 0.177, 0.169, 0.162, 0.155, 0.1495, 0.144, 0.139, 0.134, 0.130, 0.126, 0.122, 0.1185, 0.115, 0.112, 0.109]


def calc_energy_alum(en):
    return 1.50628 * 10**(-4) * en**4 - 3.33884 * 10**(-3) * en**3 + 3.06264 * 10**(-2) * en**2 - 0.152662 * en + 0.488032


def calc_energy_mylar(en):
    return 2.38081 * 10**(-4) * en**4 - 4.91836 * 10**(-3) * en**3 + 4.08927 * 10**(-2) * en**2 - 0.176174 * en + 0.443130


# coefficients, highest power first ; result in MeV/μm, E in MeV
p_al = [1.50628e-4, -3.33884e-3, 3.06264e-2, -0.152662, 0.488032]
p_mylar = [2.38081e-4, -4.91836e-3, 4.08927e-2, -0.176174, 0.443130]


def S_al(E): return np.polyval(p_al, E)
def S_mylar(E): return np.polyval(p_mylar, E)


#E = np.linspace(2.4, 5.6, 300)
#plt.plot(E, S_al(E)*1000, label="Aluminium")
#plt.plot(E, S_mylar(E)*1000, label="Mylar")
#plt.xlabel("Alpha energy (MeV)")
#plt.ylabel("Stopping power −dE/dx (keV/μm)")
#plt.legend(); plt.grid(alpha=0.3)
#plt.show()

energy_target = det_E_no_obs - det_E_with_mylar  # MeV for mylar
count = 1
while energy_target >= 0:
    de = -calc_energy_mylar(energy_target)
    energy_target += de
    print(energy_target)
    count += 1
print(f'\nMethod2 Width of mylar: {count} μm')

energy_target = det_E_no_obs - det_E_with_alum  # MeV for alum
count = 0
while energy_target >= 0:
    de = -calc_energy_alum(energy_target)
    energy_target += de
    count += 1
print(f'\nMethod2 Width of alum: {count} μm')
# ---------------------------------------------------------------------------
# Error on the obstacle widths
# ---------------------------------------------------------------------------
# Sources of uncertainty:
#   1) peak position:  sigma_P = FWHM / 2.355  (from calculate_fwhm of each spectrum)
#   2) calibration line E = A_CAL*P + B_CAL: uncertainty of a and b (with covariance)
FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))   # 1/2.355
A_CAL, B_CAL = 0.000772, 1.7152                    # same numbers used above

# Uncertainty of the calibration line from the radium fit (residual scatter, n-2 dof)
Pc = np.array(Emax, dtype=float)
Ec = np.array(known_energies_MeV, dtype=float)
n_cal = len(Pc)
Sxx = np.sum((Pc - Pc.mean()) ** 2)
s2 = np.sum((Ec - (a * Pc + b)) ** 2) / (n_cal - 2)
var_a = s2 / Sxx
var_b = s2 * (1 / n_cal + Pc.mean() ** 2 / Sxx)
cov_ab = -s2 * Pc.mean() / Sxx
print(f"\nCalibration: sigma_a = {np.sqrt(var_a):.2e} MeV/ch, sigma_b = {np.sqrt(var_b):.4f} MeV")

# Peak position and its uncertainty for each spectrum
files = {'none': 'cal_am_5cm_300s.txt',
         'mylar': 'spectrum_am_5cm_mylar_300s.txt',
         'alum': 'spectrum_am_5cm_alum_300s.txt'}
P = {k: max_pulse(f) for k, f in files.items()}
fwhm = {k: measure_fwhm(read_spectrum(f), P[k]) for k, f in files.items()}
sP = {k: v * FWHM_TO_SIGMA for k, v in fwhm.items()}
for k in files:
    print(f"{k:>6}: peak = {P[k]:.0f}, FWHM = {fwhm[k]:.1f}, sigma_P = {sP[k]:.1f}")


def energy_with_error(pulse, sig_pulse):
    '''E = a*P + b ; var = a^2 sP^2 + P^2 var_a + var_b + 2 P cov_ab'''
    E = A_CAL * pulse + B_CAL
    var = (A_CAL * sig_pulse) ** 2 + pulse ** 2 * var_a + var_b + 2 * pulse * cov_ab
    return E, np.sqrt(var)


# --- Method 1: dx = (E_at_target - E_det) / (dE/dx) ---------------------------
print("\n--- Method 1 width error ---")
for name, key, dedx, dx_val in [('mylar', 'mylar', dedx_mylar, dx_mylar),
                                ('alum', 'alum', dedx_alum, dx_alum)]:
    E_det, sE = energy_with_error(P[key], sP[key])
    sdx = sE / dedx
    print(f"{name}: E_det = {E_det:.3f} ± {sE:.3f} MeV  ->  width = {dx_val:.2f} ± {sdx:.2f} μm")


# --- Method 2: iterative stopping-power loop (Monte Carlo on the energy loss) --
def method2_width(delta_E, stopping, max_iter=10000):
    e = delta_E
    n_steps = 0
    while e >= 0.00001 and n_steps < max_iter:
        e += -stopping(e)
        n_steps += 1
    return n_steps - 1


rng = np.random.default_rng(0)
print("\n--- Method 2 width error (Monte Carlo, 5000 trials) ---")
for name, key, stopping in [('mylar', 'mylar', calc_energy_mylar),
                            ('alum', 'alum', calc_energy_alum)]:
    dE = det_E_no_obs - (det_E_with_mylar if key == 'mylar' else det_E_with_alum)
    # b cancels in the difference: var = a^2 (sP0^2 + sP1^2) + (P0-P1)^2 var_a
    sdE = np.sqrt(A_CAL ** 2 * (sP['none'] ** 2 + sP[key] ** 2)
                  + (P['none'] - P[key]) ** 2 * var_a)
    w0 = method2_width(dE, stopping)
    ws = [method2_width(rng.normal(dE, sdE), stopping) for _ in range(5000)]
    print(f"{name}: dE = {dE:.3f} ± {sdE:.3f} MeV  ->  width = {w0} ± {np.std(ws, ddof=1):.2f} μm")

print('\nfin\n')