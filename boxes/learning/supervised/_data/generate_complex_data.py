import numpy as np
import matplotlib.pyplot as plt

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/LastBlackBox'
box_path = repo_path + '/boxes/learning'
data_path = box_path + '/supervised/_data/complex.csv'

# Generate X
extent = 7
num_samples = 1000
x = (np.random.rand(num_samples) * extent) - (extent/2)

# Inital guesses
A = 0.75
B = 0.05
C = -8.5
D = 0.21
E = 9.15
params = np.array([A,B,C,D,E])
num_params = len(params)

# Define function
def func(x, params):
    A = params[0] 
    B = params[1] 
    C = params[2] 
    D = params[3] 
    E = params[4] 
    return A * (x**5) + B * (x**4) + C * (x**3) + D * (x**2) + E * (x)

# Compute Y
y = func(x, params)

# Add noise
noise = (np.random.randn(num_samples) - 0.5)*3
y = y + noise

# Plot
plt.plot(x,y, '.', markersize=1)
plt.show()

# Save
data = np.vstack((x, y)).T
np.savetxt(data_path, data, fmt='%.7f', delimiter=',')

#FIN