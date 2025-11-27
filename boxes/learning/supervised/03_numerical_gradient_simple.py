# Small steps down numerically computed gradient to find model for simple data
import os
import numpy as np
import matplotlib.pyplot as plt
import LBB.config as Config

# Specify paths
box_path = f"{Config.repo_path}/boxes/learning"
data_path = f"{box_path}/supervised/_data/simple.csv"
this_path = os.path.basename(__file__)
output_path = f"{box_path}/supervised/my_{this_path[:-3]}.png"

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]

# Define function
def func(x, params):
    A = params[0] 
    B = params[1] 
    return (A * x) + B

# Initial guesses for params (mean=0, sigma=1)
A = np.random.randn(1)[0]
B = np.random.randn(1)[0]
params = np.array([A,B])
num_params = len(params)
    
# Define loss (mean squared error)
def loss(x, y, params):
    guess = func(x, params)
    err = y - guess
    return np.mean(err*err)

# Train
alpha = 1                                           # Learning rate
step_size = 0.0005
num_steps = 20000
report_interval = 1000
initial_loss = loss(x, y, params)
losses = [initial_loss]
for i in range(num_steps):
    current_params = np.copy(params)                # Copy current parameters
    current_loss = loss(x, y, params)               # Measure current loss
    for p in range(num_params):                     # Update each parameter
        guess_params = np.copy(current_params)      # Reset parameter guesses
        guess_params[p] += step_size                # Increment one parameter
        new_loss = loss(x, y, guess_params)         # Measure loss
        grad = current_loss - new_loss              # Measure numerical gradient
        params[p] += (grad * alpha)                 # Update parameter
    
    # Store loss
    final_loss = loss(x, y, params)                 # Measure final loss
    losses.append(final_loss)

    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("{0}: MSE: {1:.2f}, Params: {2}".format(i, final_loss, params))

# Compare prediction to data
prediction = func(x, params)

plt.figure(figsize=(8,4), dpi=150)
plt.suptitle(f"{this_path[3:-3]}")
plt.subplot(1,2,1)
plt.plot(x, y, 'b.', markersize=1)              # Plot data
plt.plot(x, prediction, 'r.', markersize=1)     # Plot prediction
plt.xlabel("X (input)")
plt.ylabel("Y (output)")
plt.subplot(1,2,2)
plt.plot(np.array(losses))                      # Plot loss over training
plt.xlabel("X (input)")
plt.ylabel("Loss")
plt.savefig(output_path)

#FIN
