# Train a multi-layer perceptron to reproduce the underlying function for complex (nonlinear) data
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, nn
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

# Define network (size of hidden layer)
num_hidden_neurons = 10

# Initialize hidden layer (size: num_hidden_neurons)
W1 = np.random.rand(num_hidden_neurons)
B1 = np.random.rand(num_hidden_neurons)
W1 = np.expand_dims(W1,0)
B1 = np.expand_dims(B1,0)

# Initialize output layer (size: num_hidden_neurons)
W2 = np.random.rand(num_hidden_neurons)
B2 = np.random.rand(num_hidden_neurons)
W2 = np.expand_dims(W2,0)
B2 = np.expand_dims(B2,0)

# Define function (network)
def func(x, W1, B1, W2, B2):
    hidden = x.dot(W1) + B1
    activations = nn.sigmoid(hidden)
    interim = activations.dot(W2.T) + B2
    output = jnp.sum(interim, axis=1)
    return output

# Define loss (mean squared error)
def loss(x, y, W1, B1, W2, B2):
    guess = func(x, W1, B1, W2, B2)
    err = np.squeeze(y) - guess
    return jnp.mean(err*err)

# Compute gradient (w.r.t. parameters)
grad_loss_W1 = jit(grad(loss, argnums=2))
grad_loss_B1 = jit(grad(loss, argnums=3))
grad_loss_W2 = jit(grad(loss, argnums=4))
grad_loss_B2 = jit(grad(loss, argnums=5))

# Train
alpha = .0001
beta = 0.99
deltas_W1 = np.zeros(num_hidden_neurons)
deltas_B1 = np.zeros(num_hidden_neurons)
deltas_W2 = np.zeros(num_hidden_neurons)
deltas_B2 = np.zeros(num_hidden_neurons)
report_interval = 1000
num_steps = 10000
initial_loss = loss(x, y, W1, B1, W2, B2)
losses = [initial_loss]
for i in range(num_steps):    

    # Compute gradients
    gradients_W1 = grad_loss_W1(x, y, W1, B1, W2, B2)
    gradients_B1 = grad_loss_B1(x, y, W1, B1, W2, B2)
    gradients_W2 = grad_loss_W2(x, y, W1, B1, W2, B2)
    gradients_B2 = grad_loss_B2(x, y, W1, B1, W2, B2)

    # Update deltas
    deltas_W1 = (alpha * gradients_W1) + (beta * deltas_W1)
    deltas_B1 = (alpha * gradients_B1) + (beta * deltas_B1)
    deltas_W2 = (alpha * gradients_W2) + (beta * deltas_W2)
    deltas_B2 = (alpha * gradients_B2) + (beta * deltas_B2)

    # Update parameters
    W1 -= (deltas_W1)
    B1 -= (deltas_B1)
    W2 -= (deltas_W2)
    B2 -= (deltas_B2)

    # Store loss
    final_loss = loss(x, y, W1, B1, W2, B2)
    losses.append(final_loss)

    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("{0}: MSE: {1:.2f}".format(i, final_loss))

# Compare prediction to data
prediction = func(x, W1, B1, W2, B2)

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
