import numpy as np
import matplotlib.pyplot as plt
import time

# Get user name
import os
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
box_path = repo_path + '/boxes/learning'
data_path = box_path + '/supervised/_data/simple.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]

# Define function
def func(x, params):
    A = params[0] 
    B = params[1] 
    return (A * x) + B

# Inital guesses
A = np.random.rand(1)[0] - 0.5
B = np.random.rand(1)[0] - 0.5
params = np.array([A,B])
num_params = len(params)
    
# Define loss (mean squared error)
def loss(x, y, params):
    guess = func(x, params)
    err = y - guess
    return np.mean(err*err)

# Define gradient
def grad(x, y, params):
    A = params[0]
    B = params[1] 
    dE_dA = 2 * np.mean(-x * (y - ((A * x) + B)))
    dE_dB = 2 * np.mean(-(y - ((A * x) + B)))
    return np.array([dE_dA, dE_dB])

# Train
alpha = .001                        # Learning rate
num_steps = 20000
report_interval = 100
initial_loss = loss(x, y, params)
losses = [initial_loss]
for i in range(num_steps):
    gradients = grad(x, y, params)      # Compute gradients for each paramter
    params -= (gradients * alpha)       # Update parameters

    # Store loss
    final_loss = loss(x, y, params)     # Compute loss
    losses.append(final_loss)
    
    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("MSE: {0:.2f}, Params: {1}".format(final_loss, params))

# Compare prediction to data
prediction = func(x, params)
plt.subplot(1,2,1)
plt.plot(x, y, 'b.', markersize=1)              # Plot data
plt.plot(x, prediction, 'r.', markersize=1)     # Plot prediction
plt.subplot(1,2,2)
plt.plot(np.array(losses))                      # Plot loss over training
plt.show()

#FIN
