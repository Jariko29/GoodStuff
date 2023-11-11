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

# Create bar chart
plt.bar(x, y, color='blue',width = 0.088, edgecolor='black')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Bar Chart')
plt.savefig('barchart.png')
print('Done')
quit()