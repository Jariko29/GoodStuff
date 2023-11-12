import matplotlib.pyplot as plt
import scipy.stats as stats

with open('metrisis.txt', 'r') as f:
    data = f.readlines()

x = []
y = []
for line in data:
    values = line.split()
    x.append(float(values[0]))
    y.append(float(values[1]))

maxwell = stats.maxwell
params = maxwell.fit(values, floc=0)
print(params)

plt.bar(x, y, color='blue',width = 0.088, edgecolor='black')
plt.show()
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Bar Chart')
plt.savefig('barchart.png')
print('Done')
quit()

