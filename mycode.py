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

# Create a new figure with a specific size (width, height)
plt.figure(figsize=(10, 6))

# Create bar chart
plt.bar(x, y, color='blue', width=0.1)

# Add grid lines
plt.grid(True)

# Set labels and title with a larger font size
plt.xlabel('X Axis', fontsize=14)
plt.ylabel('Y Axis', fontsize=14)
plt.title('Bar Chart', fontsize=16)

# Rotate x-axis labels if they overlap
plt.xticks(rotation=45)

# Save the figure
plt.savefig('barchart.png', bbox_inches='tight')

print('Done')
quit()