# Load simple data
import os
import numpy as np
import matplotlib.pyplot as plt
import LBB.config as Config

# Specify paths
box_path = f"{Config.repo_path}/boxes/learning"
data_path = f"{box_path}/supervised/_data/complex.csv"
this_path = os.path.basename(__file__)
output_path = f"{box_path}/supervised/my_{this_path[:-3]}.png"

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]

# Plot data
plt.figure(figsize=(6,4), dpi=150)
plt.title(f"{this_path[3:-3]}")
plt.plot(x,y,'.', alpha=0.5)
plt.xlabel("X (input)")
plt.ylabel("Y (output)")
plt.savefig(output_path)

#FIN
