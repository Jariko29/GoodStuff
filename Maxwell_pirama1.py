import matplotlib.pyplot as plt

with open('metrisis.txt', 'r') as f:
    data = f.readlines()

x = []
y = []
for line in data:
    values = line.split()
    x.append(float(values[0]))
    y.append(float(values[1]))

plt.bar(x, y, color='blue',width = 0.088, edgecolor='black')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Bar Chart')
plt.savefig('barchart.png')
print('Done')
quit()

