import matplotlib.pyplot as plt

# Read data from file
with open('metrisis.txt', 'r') as f:
    data = f.readlines()

# Split data into x and y values
x = []
y = []
for line in data:
    values = line.split()
    x.append(float(values[0]))
    y.append(float(values[1]))

print(x)
# Create histogram
plt.hist(x, bins=24, weights=[0.09] * len(x))
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Histogram')
plt.savefig('histogram.png')
print('Done')
quit()