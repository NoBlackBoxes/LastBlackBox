# Generate Complex Data
import numpy as np
import matplotlib.pyplot as plt
import LBB.config as Config

# Specify paths
box_path = f"{Config.repo_path}/boxes/learning"
data_path = f"{box_path}/supervised/_data/complex.csv"

# Generate X (inputs)
extent = 7
num_samples = 1000
x = (np.random.rand(num_samples) * extent) - (extent/2)

# Hidden parameters
A = 0.75
B = 0.05
C = -8.5
D = 0.21
E = 9.15
params = np.array([A,B,C,D,E])
num_params = len(params)

# Define function (which we will try to learn/discover)
def func(x, params):
    A = params[0]
    B = params[1]
    C = params[2]
    D = params[3]
    E = params[4]
    return A * (x**5) + B * (x**4) + C * (x**3) + D * (x**2) + E * (x)

# Compute Y (outputs)
y = func(x, params)

# Add noise ("normally distributed", mean=0, sigma=1)
noise = np.random.randn(num_samples) * 3.0
y = y + noise

# Save
data = np.vstack((x, y)).T
np.savetxt(data_path, data, fmt='%.7f', delimiter=',')

#FIN