import matplotlib.pyplot as plt

# Create a new figure
plt.figure()

# Plot a simple line
plt.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])

# Add a title
plt.title('Hello, World!')

# Add labels to the axes
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Show the plot
plt.show()