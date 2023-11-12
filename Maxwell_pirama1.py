import matplotlib.pyplot as plt
from scipy.stats import maxwell

with open('metrisis.txt', 'r') as f:
    data = f.readlines()

x = []
y = []
for line in data:
    values = line.split()
    x.append(float(values[0]))
    y.append(float(values[1]))

params = maxwell.fit(y, floc=0)
print(params)
R = maxwell.pdf(x, *params)  
print(R)

plt.bar(x, y, color='blue',width = 0.088, edgecolor='black')
plt.plot(x, R, color='red', linewidth=1)
plt.xlabel('Ταχύτητα (m/s)')
plt.ylabel('Συχνότητα (μπάλες/s)')
plt.title('Ιστόγραμμα Κατανομής Ταχυτήτων-Maxwell')
plt.savefig('barchart.png')
print('Done')
quit()
