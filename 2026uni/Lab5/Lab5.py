from pathlib import Path
from math import sqrt

import matplotlib.pyplot as plt


RYDBERG_EV = 13.6
TARGETS = (
	("Mo", 42),
	("Ag", 47),
	("BaCl2 (Ba)", 56),
	("SrSO4 (Sr)", 38),
	("I2 (I)", 53),
)


def weighted_line_fit(x_values, y_values, uncertainties):
	weights = [1 / uncertainty**2 for uncertainty in uncertainties]
	sum_weights = sum(weights)
	sum_weighted_x = sum(w * x for w, x in zip(weights, x_values))
	sum_weighted_y = sum(w * y for w, y in zip(weights, y_values))
	sum_weighted_x2 = sum(w * x**2 for w, x in zip(weights, x_values))
	sum_weighted_xy = sum(w * x * y for w, x, y in zip(weights, x_values, y_values))

	determinant = sum_weights * sum_weighted_x2 - sum_weighted_x**2
	if determinant <= 0:
		raise ValueError("The measurement points do not determine a line.")

	slope = (sum_weights * sum_weighted_xy - sum_weighted_x * sum_weighted_y) / determinant
	intercept = (sum_weighted_x2 * sum_weighted_y - sum_weighted_x * sum_weighted_xy) / determinant
	slope_error = sqrt(sum_weights / determinant)
	intercept_error = sqrt(sum_weighted_x2 / determinant)
	return slope, intercept, slope_error, intercept_error


def main():
	# Replace each 0.0 with the measured K-alpha energy in keV, in target order.
	measured_energies = [
		0.0,  # Mo
		0.0,  # Ag
		0.0,  # BaCl2
		0.0,  # SrSO4
		0.0,  # I2
	]
	# Enter the matching energy uncertainties in keV, in the same target order.
	energy_uncertainties = [
		0.0,  # Mo uncertainty
		0.0,  # Ag uncertainty
		0.0,  # BaCl2 uncertainty
		0.0,  # SrSO4 uncertainty
		0.0,  # I2 uncertainty
	]
	if len(measured_energies) != len(TARGETS) or len(energy_uncertainties) != len(TARGETS):
		raise ValueError("Enter one energy and uncertainty for every target.")
	if any(value <= 0 for value in measured_energies + energy_uncertainties):
		raise ValueError("Replace every 0.0 placeholder with a positive measured value.")

	atomic_numbers = [atomic_number for _, atomic_number in TARGETS]
	x_values = [(atomic_number - 1) ** 2 for atomic_number in atomic_numbers]
	theoretical_energies = [RYDBERG_EV * 0.75 * x / 1000 for x in x_values]
	slope, intercept, slope_error, intercept_error = weighted_line_fit(
		x_values, measured_energies, energy_uncertainties
	)

	print("\nTarget       Z      (Z-1)^2   Theory (keV)   Measured (keV)")
	for (target, atomic_number), x, theory, energy, uncertainty in zip(
		TARGETS, x_values, theoretical_energies, measured_energies, energy_uncertainties
	):
		print(
			f"{target:<12} {atomic_number:>2} {x:>11}"
			f" {theory:>13.3f}   {energy:.3f} +/- {uncertainty:.3f}"
		)

	rydberg_ev = slope * 1000 / 0.75
	rydberg_error_ev = slope_error * 1000 / 0.75
	print(
		f"\nWeighted fit: E = ({slope:.6f} +/- {slope_error:.6f})"
		f" * (Z-1)^2 + ({intercept:.3f} +/- {intercept_error:.3f}) keV"
	)
	print(f"Experimental Rydberg constant: {rydberg_ev:.3f} +/- {rydberg_error_ev:.3f} eV")
	print(f"Accepted value used by the handout: {RYDBERG_EV:.1f} eV")

	line_x = [min(x_values), max(x_values)]
	line_y = [slope * x + intercept for x in line_x]
	plt.errorbar(
		x_values,
		measured_energies,
		yerr=energy_uncertainties,
		fmt="o",
		capsize=4,
		label="Measurements",
	)
	plt.plot(line_x, line_y, label="Weighted linear fit")
	plt.xlabel("(Z - 1)^2")
	plt.ylabel("K-alpha energy (keV)")
	plt.title("Moseley's law")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(Path(__file__).with_name("Lab5_moseley.png"), dpi=300)
	plt.show()


if __name__ == "__main__":
	main()
