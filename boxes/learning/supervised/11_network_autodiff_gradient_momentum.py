import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, nn

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/LastBlackBox'
box_path = repo_path + '/boxes/learning'
data_path = box_path + '/supervised/_data/complex.csv'

# Load data
data = np.genfromtxt(data_path, delimiter=',')
x = data[:,0]
y = data[:,1]
x = np.expand_dims(x,1) # Add dimension
y = np.expand_dims(y,1) # Add dimension

# Define network
num_inputs = 1
num_outputs = 1
num_hidden_layers = 1
num_neurons_per_layer = [num_inputs, 10, num_outputs]

# Initialize parameters (weights and biases)
weights = []
biases = []
deltas_weights = []
deltas_biases = []
for i in range(1, num_hidden_layers + num_outputs + 1):     # Skip inputs, include outputs
    w = np.random.rand(num_neurons_per_layer[i] * num_neurons_per_layer[i-1]) - 0.5
    w = np.expand_dims(w, 0)
    b = np.random.rand(num_neurons_per_layer[i] * num_neurons_per_layer[i-1]) - 0.5
    b = np.expand_dims(b, 0)
    d = np.zeros(num_neurons_per_layer[i] * num_neurons_per_layer[i-1])
    weights.append(w)
    biases.append(b)
    deltas_weights.append(d)
    deltas_biases.append(d)

# Define function (perceptron)
def func(x, weights, biases):

    num_layers = len(weights)
    num_hidden_layers = num_layers - 1
    output_layer = num_hidden_layers

    # Set input layer
    input = x

    # Hidden layers
    for i in range(num_hidden_layers):
        interim = input.dot(weights[i]) + biases[i]
        activations = nn.sigmoid(jnp.sum(interim, axis=1))
        output = jnp.expand_dims(activations,1)
        input = output
    
    # Output layer
    interim = output.dot(weights[output_layer]) + biases[output_layer]
    output = jnp.sum(interim, axis=1)
    return output

# Define loss (mean squared error)
def loss(x, y, weights, biases):
    guess = func(x, weights, biases)
    err = jnp.squeeze(y) - guess
    return jnp.mean(err*err)

# Compute gradient (w.r.t. parameters)
grad_loss_weights = jit(grad(loss, argnums=2))
grad_loss_biases = jit(grad(loss, argnums=3))

# Train
alpha = .00001
beta = 0.99
report_interval = 100
num_steps = 3000
losses = []
initial_loss = loss(x, y, weights, biases)
losses = [initial_loss]
for i in range(num_steps):    

    # Compute gradients
    gradients_weights = grad_loss_weights(x, y, weights, biases)                            # (weights)
    gradients_biases = grad_loss_biases(x, y, weights, biases)                              # (biases)

    # Update deltas
    for j in range(num_hidden_layers + num_outputs):
        deltas_weights[j] = (alpha * gradients_weights[j]) + (beta * deltas_weights[j])     # (weights)
        deltas_biases[j] = (alpha * gradients_biases[j]) + (beta * deltas_biases[j])        # (biases)

    # Update parameters
    for j in range(num_hidden_layers + num_outputs):
        weights[j] -= (deltas_weights[j])                                                   # (weights)
        biases[j] -= (deltas_biases[j])                                                     # (biases)

    # Store loss
    final_loss = loss(x, y, weights, biases)
    losses.append(final_loss)

    # Report?
    if((i % report_interval) == 0):
        np.set_printoptions(precision=3)
        print("MSE: {0:.2f}".format(final_loss))

# Compare prediction to data
prediction = func(x, weights, biases)
plt.subplot(1,2,1)
plt.plot(x, y, 'b.', markersize=1)              # Plot data
plt.plot(x, prediction, 'r.', markersize=1)     # Plot prediction
plt.subplot(1,2,2)
plt.plot(np.array(losses))                      # Plot loss over training
plt.show()

#FIN
