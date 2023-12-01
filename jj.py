import math

# Δεδομένα
mass_astronaut = 80  # Μάζα του αστροναύτη σε κιλά
distance = 5.0  # Απόσταση που απομακρύνθηκε ο αστροναύτης από τον σταθμό σε μέτρα
power_laser = 1000  # Ισχύς του laser σε watt
oxygen_remaining = 10  # Υπόλοιπο οξυγόνου σε ώρες

# Σταθερές
speed_of_light = 3.0e8  # Ταχύτητα του φωτός σε μέτρα/δευτερόλεπτο

# Υπολογισμοί
acceleration = (2 * power_laser) / (speed_of_light * mass_astronaut)
time_to_return = math.sqrt(2 * distance / acceleration)

# Έλεγχος αν ο χρόνος υπερβαίνει το υπόλοιπο οξυγόνου
if time_to_return > oxygen_remaining:
    print("Δεν είναι δυνατή η επιστροφή. Ο χρόνος υπερβαίνει το υπόλοιπο οξυγόνου.")
else:
    print(f"Ο αστροναύτης θα χρειαστεί περίπου {time_to_return:.2f} δευτερόλεπτα για να επιστρέψει.")
