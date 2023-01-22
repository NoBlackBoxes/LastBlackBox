import numpy as np
import matplotlib.pyplot as plt

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/LastBlackBox'
box_path = repo_path + '/boxes/learning'
data_path = box_path + '/supervised/_data/simple.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]

# Plot data
plt.plot(x,y,'.', alpha=0.5)
plt.show()

#FIN
