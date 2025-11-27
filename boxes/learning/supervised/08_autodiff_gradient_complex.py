# Small steps down analytically computed gradient to find model for complex data
# - We use auto-diff (via jax) to compute the gradient
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit
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

# Define function
def func(x, params):
    A = params[0] 
    B = params[1] 
    C = params[2] 
    D = params[3] 
    E = params[4] 
    return A * (x**5) + B * (x**4) + C * (x**3) + D * (x**2) + E * (x)
    
# Initial guesses for params (mean=0, sigma=1)
A = np.random.randn(1)[0]
B = np.random.randn(1)[0]
C = np.random.randn(1)[0]
D = np.random.randn(1)[0]
E = np.random.randn(1)[0]
params = np.array([A,B,C,D,E])
num_params = len(params)
    
# Define loss (mean squared error)
def loss(x, y, params):
    guess = func(x, params)
    err = y - guess
    return np.mean(err*err)

# Compute gradient (w.r.t. parameters)
grad_loss = grad(loss, argnums=2)       # Use Jax to compute gradient
grad_loss_jit = jit(grad_loss)          # Compile gradient function

# Train
alpha = .00001
num_steps = 20000
report_interval = 1000
initial_loss = loss(x, y, params)
losses = [initial_loss]
for i in range(num_steps):
    # gradients = grad_loss(x, y, params)       # Compute gradient (uncompiled)
    gradients = grad_loss_jit(x, y, params)     # Compute gradient (compiled)
    params -= (gradients * alpha)               # Update parameters

    # Store loss
    final_loss = loss(x, y, params)             # Measure final loss
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
plt.plot(np.log(np.array(losses)))              # Plot log loss over training
plt.xlabel("X (input)")
plt.ylabel("Loss (log)")
plt.savefig(output_path)

#FIN
