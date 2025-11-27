# Train a perceptron to reproduce the underlying function for complex (nonlinear) data
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
x = np.expand_dims(x,1) # Add dimension
y = np.expand_dims(y,1) # Add dimension

# Define network
num_neurons = 10

# Initialize weights and biases
weights = np.random.randn(num_neurons)
biases = np.random.randn(num_neurons)
weights = np.expand_dims(weights,0) # Add dimension
biases = np.expand_dims(biases,0)   # Add dimension
        
# Define function (perceptron)
def func(x, weights, biases):
    interim = jnp.dot(x, weights) + biases
    output = jnp.sum(interim, axis=1)
    return output

# Define loss (mean squared error)
def loss(x, y, weights, biases):
    guess = func(x, weights, biases)
    err = np.squeeze(y) - guess
    return jnp.mean(err*err)

# Compute gradient (w.r.t. parameters)
grad_loss_weights = jit(grad(loss, argnums=2))
grad_loss_biases = jit(grad(loss, argnums=3))

# Train
alpha = .001
beta = 0.99
deltas_weights = np.zeros(num_neurons)
deltas_biases = np.zeros(num_neurons)
report_interval = 100
num_steps = 3000
initial_loss = loss(x, y, weights, biases)
losses = [initial_loss]
for i in range(num_steps):    

    # Compute gradients
    gradients_weights = grad_loss_weights(x, y, weights, biases)            # (weights)
    gradients_biases = grad_loss_biases(x, y, weights, biases)              # (biases)

    # Update deltas
    deltas_weights = (alpha * gradients_weights) + (beta * deltas_weights)  # (weights)
    deltas_biases = (alpha * gradients_biases) + (beta * deltas_biases)     # (biases)

    # Update parameters
    weights -= (deltas_weights)                                             # (weights)
    biases -= (deltas_biases)                                               # (biases)

    # Store loss
    final_loss = loss(x, y, weights, biases)
    losses.append(final_loss)

    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("{0}: MSE: {1:.2f}, Weights: {2}".format(i, final_loss, weights))

# Compare prediction to data
prediction = func(x, weights, biases)

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
