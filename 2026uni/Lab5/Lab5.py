from pathlib import Path

import matplotlib.pyplot as plt


AM_ENERGY_KEV = 59.5
CS_LOW_ENERGY_KEV = 32.0
CS_LOW_PEAK_MAX_PULSE_HEIGHT = 200.0


def read_spectrum(filename: str) -> list[tuple[float, int]]:
    spectrum = []
    with Path(__file__).with_name(filename).open() as file:
        next(file, None)
        for line in file:
            pulse_height, count = line.split()
            spectrum.append((float(pulse_height), int(count)))
    if not spectrum:
        raise ValueError(f"No spectrum data found in {filename}.")
    return spectrum


def main() -> None:
    am_peak = max(
        read_spectrum("spectrum_am_600s.txt"),
        key=lambda point: point[1],
    )
    cs_low_peak = max(
        (
            point
            for point in read_spectrum("spectrum_cs_600s.txt")
            if point[0] < CS_LOW_PEAK_MAX_PULSE_HEIGHT
        ),
        key=lambda point: point[1],
    )

    slope = (AM_ENERGY_KEV - CS_LOW_ENERGY_KEV) / (
        am_peak[0] - cs_low_peak[0]
    )
    intercept = CS_LOW_ENERGY_KEV - slope * cs_low_peak[0]

    print(
        f"Calibration line: E = {slope:.6f} * pulse height "
        f"+ {intercept:.3f} keV"
    )
    print(
        f"Cs low-energy peak: {cs_low_peak[0]:.2f} "
        f"(~{CS_LOW_ENERGY_KEV:.1f} keV)"
    )
    print(f"Am peak: {am_peak[0]:.2f} ({AM_ENERGY_KEV:.1f} keV)")

    line_x = [0.0, am_peak[0]]
    line_y = [slope * pulse_height + intercept for pulse_height in line_x]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [cs_low_peak[0], am_peak[0]],
        [CS_LOW_ENERGY_KEV, AM_ENERGY_KEV],
        color="#287c78",
        edgecolor="black",
        zorder=3,
        label="Calibration peaks",
    )
    plt.annotate(
        "Cs low-energy peak (~32 keV)",
        (cs_low_peak[0], CS_LOW_ENERGY_KEV),
        xytext=(6, 6),
        textcoords="offset points",
    )
    plt.annotate(
        "Am-241 peak (59.5 keV)",
        (am_peak[0], AM_ENERGY_KEV),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
    )
    plt.plot(
        line_x,
        line_y,
        color="#c41c1c",
        label=f"E = {slope:.6f}P + {intercept:.3f}",
    )
    plt.xlabel("Pulse height")
    plt.ylabel("Energy (keV)")
    plt.title("Am/Cs Energy Calibration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output = Path(__file__).with_name("Lab5_calibration.png")
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved calibration graph to {output}")


if __name__ == "__main__":
    main()