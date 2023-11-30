import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_values = np.array([0.1, 0.15, 0.2, 0.25])
y_values = np.array([-110.4, -120.0, -177.4, -161.2])

# Assuming you have confidence interval values for each y-value
confidence_interval = np.array([2.5, 3.8, 5.6, 7.2])

# Calculate upper and lower bounds for the confidence interval
upper_bound = y_values + confidence_interval
lower_bound = y_values - confidence_interval

# Create a line plot
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Add a filled confidence interval
plt.fill_between(x_values, lower_bound, upper_bound, alpha=0.2)

# Add labels and title
plt.xlabel('Learning rate')
plt.ylabel('Avg total reward (50 episodes)')
plt.title('Avg total reward as function of learning rate')


# Show the plot
plt.show()


# Sample data
x_values = np.array([0.1, 0.15, 0.2, 0.25])
y_values = np.array([-110.4, -116.7, -118.1, -144.1])

# Assuming you have confidence interval values for each y-value
confidence_interval = np.array([2.5, 4.1, 1.3, 9.7])

# Calculate upper and lower bounds for the confidence interval
upper_bound = y_values + confidence_interval
lower_bound = y_values - confidence_interval

# Create a line plot
plt.plot(x_values, y_values, marker='o', linestyle='-', color='orange')

# Add a filled confidence interval
plt.fill_between(x_values, lower_bound, upper_bound, alpha=0.2, color='orange')

# Add labels and title
plt.xlabel('Eligibility trace')
plt.ylabel('Avg total reward (50 episodes)')
plt.title('Avg total reward as function of eligibility trace')


# Show the plot
plt.show()


# Sample data
x_values = np.array(range(1,31))
y_values = np.array([0 for _ in range(1,15)]+[1 for _ in range(15, 31)])

# Create a line plot
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Time horizon')
plt.ylabel('Probability')
plt.title('Probability of exiting maze')


# Show the plot
plt.show()
