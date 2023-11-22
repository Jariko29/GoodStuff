import math

# Δεδομένα
Vm = 150  # Μέγιστη τάση
R = 40.0  # Αντίσταση
L = 80.0e-3  # Πηνίο αυτεπαγωγής
C = 50.0e-6  # Πυκνωτής
frequency = 100  # Συχνότητα

# Represent the voltages as complex numbers
VR0_complex = VR0
VL0_complex = VL0 * (1j)  # The voltage across the inductor leads by 90 degrees
VC0_complex = VC0 * (-1j)  # The voltage across the capacitor lags by 90 degrees

# Add the complex voltages
V_total_complex = VR0_complex + VL0_complex + VC0_complex

# The magnitude of the total voltage is the absolute value of the complex total voltage
V_total = abs(V_total_complex)

# Now you can check if V_total is close to Vm
if math.isclose(V_total, Vm, rel_tol=1e-9):
    print("Οι τάσεις είναι περίπου ίσες.")
else:
    print("Οι τάσεις δεν είναι ίσες.")


# Υπολογισμοί
Im = Vm / (R * math.sqrt(2))  # Ρευματική ένταση
omega = 2 * math.pi * frequency  # Γωνιακή συχνότητα

# Πτώση τάσης στην αντίσταση (VR0)
VR0 = Im * R

# Πτώση τάσης στο πηνίο (VL0)
VL0 = Im * omega * L

# Πτώση τάσης στον πυκνωτή (VC0)
VC0 = Im / (omega * C)

# Ελέγχουμε εάν η συνολική τάση είναι περίπου ίση με την αρχική τάση
if math.isclose(VR0 + VL0 + VC0, Vm, rel_tol=1e-9):
    print("Οι τάσεις είναι περίπου ίσες.")
else:
    print("Οι τάσεις δεν είναι ίσες.")

# Εκτύπωση αποτελεσμάτων
print(f"VR0: {VR0} V")
print(f"VL0: {VL0} V")
print(f"VC0: {VC0} V")
