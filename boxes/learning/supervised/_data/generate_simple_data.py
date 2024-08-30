import numpy as np
import matplotlib.pyplot as plt

# Get user name
import os
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
box_path = repo_path + '/boxes/learning'
data_path = box_path + '/supervised/_data/simple.csv'

# Generate X
extent = 7
num_samples = 1000
x = (np.random.rand(num_samples) * extent) - (extent/2)

# Inital guesses
A = 1.75
B = -4.22
params = np.array([A, B])
num_params = len(params)

# Define function
def func(x, params):
    A = params[0]
    B = params[1]
    return A * x + B

# Compute Y
y = func(x, params)

# Add noise
noise = (np.random.randn(num_samples) - 0.5) * 1
y = y + noise

# Plot
plt.plot(x,y, '.', markersize=1)
plt.show()

# Save
data = np.vstack((x, y)).T
np.savetxt(data_path, data, fmt='%.7f', delimiter=',')

#FIN