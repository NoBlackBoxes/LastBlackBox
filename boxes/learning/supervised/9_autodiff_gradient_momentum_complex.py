import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/LastBlackBox'
box_path = repo_path + '/boxes/learning'
data_path = box_path + '/supervised/_data/complex.csv'

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

# Compute gradient (w.r.t. parameters)
grad_loss = grad(loss, argnums=2)       # Use Jax to compute gradient
grad_loss_jit = jit(grad_loss)          # Compile gradient function

# Train
alpha = .00001
beta = 0.99
deltas = np.zeros(num_params)
num_steps = 10000
report_interval = 100
initial_loss = loss(x, y, params)
losses = [initial_loss]
for i in range(num_steps):
    # gradients = grad_loss(x, y, params)           # Compute gradient (uncompiled)
    gradients = grad_loss_jit(x, y, params)         # Compute gradient (compiled)
    deltas = (alpha * gradients) + (beta * deltas)  # Update deltas
    params -= (deltas)                              # Update parameters

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
