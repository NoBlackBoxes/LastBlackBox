import numpy as np
import matplotlib.pyplot as plt

# Get user name
import os
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
box_path = repo_path + '/boxes/learning'
data_path = box_path + '/supervised/_resources/complex.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]

# Define function
def func(x, params):
    A = params[0] 
    B = params[1] 
    C = params[2] 
    D = params[3] 
    E = params[4] 
    return A * (x**5) + B * (x**4) + C * (x**3) + D * (x**2) + E * (x)
    
# Inital guesses
A = np.random.rand(1)[0] - 0.5
B = np.random.rand(1)[0] - 0.5
C = np.random.rand(1)[0] - 0.5
D = np.random.rand(1)[0] - 0.5
E = np.random.rand(1)[0] - 0.5
params = np.array([A,B,C,D,E])
num_params = len(params)
    
# Define loss (mean squared error)
def loss(x, y, params):
    guess = func(x, params)
    err = y - guess
    return np.mean(err*err)

# Train
step_size = 0.000001
num_steps = 10000
alpha = 1
beta = 0.999
deltas = np.zeros(num_params)
report_interval = 100
initial_loss = loss(x, y, params)
losses = [initial_loss]
for i in range(num_steps):
    current_params = np.copy(params)            # Copy current parameters
    current_loss = loss(x, y, params)           # Measure current loss
    for p in range(num_params):                 # Update each parameter
        guess_params = np.copy(current_params)          # Reset parameter guesses
        guess_params[p] += step_size                    # Increment one parameter
        new_loss = loss(x, y, guess_params)             # Measure loss
        grad = current_loss - new_loss                  # Measure gradient
        deltas[p] = (alpha * grad) + (beta * deltas[p]) # Update delta
        params[p] = params[p] + deltas[p]               # Update parameter

    # Store loss
    final_loss = loss(x, y, params)             # Measure final loss
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
