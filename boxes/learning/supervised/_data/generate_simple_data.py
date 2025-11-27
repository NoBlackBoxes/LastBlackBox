# Generate Simple Data
import numpy as np
import matplotlib.pyplot as plt
import LBB.config as Config

# Specify paths
box_path = f"{Config.repo_path}/boxes/learning"
data_path = f"{box_path}/supervised/_data/simple.csv"

# Generate X (inputs)
extent = 7
num_samples = 1000
x = (np.random.rand(num_samples) * extent) - (extent/2)

# Hidden parameters
A = 1.75            # Slope
B = -4.22           # Y-intercept
params = np.array([A, B])
num_params = len(params)

# Define function (which we will try to learn/discover)
def func(x, params):
    A = params[0]
    B = params[1]
    return A * x + B

# Compute Y (outputs)
y = func(x, params)

# Add noise ("normally distributed", mean=0, sigma=1.5)
noise = np.random.randn(num_samples) * 1.5
y = y + noise

# Save
data = np.vstack((x, y)).T
np.savetxt(data_path, data, fmt='%.7f', delimiter=',')

#FIN