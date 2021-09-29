import numpy as np

class function(object): # Base class for layers
    def forward(self):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError
    def get_params(self):
        return []

class optimiser(object): # Base class for optimisers
    def __init__(self,parameters):
        self.parameters = parameters
    def step(self):
        raise NotImplementedError
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

class tensor: # Base class for storing stuff
    def __init__(self,dimensions):
        self.data = np.ndarray(dimensions,np.float32)
        self.grad = np.ndarray(dimensions,np.float32)
        self.size = dimensions

class Linear(function): ## An example of a layer
    def __init__(self,in_nodes,out_nodes):
        self.weights = tensor((in_nodes,out_nodes))
        self.bias    = tensor((1,out_nodes)) # Shifts the activation function to the left/right (changes how much input is required to produce output)
        self.type    = 'linear'
    
    def forward(self,x): ##  Calculates the forward pass of the network
        output = np.dot(x,self.weights.data) + self.bias.data # wx + b
        self.input = x
        return output

    def backward(self,d_y): ## Calculates the backwards pass of the network
        # These are the gradients of the loss with respect to the weights and bias terms
        self.weights.grad += np.dot(self.input.T,d_y) 
        self.bias.grad    += np.sum(d_y,axis=0,keepdims=True)
        # Pass back the "remaining" gradient for the next layer
        grad_input         = np.dot(d_y,self.weights.data.T)
        return grad_input

    def get_params(self):
        return [self.weights,self.bias]
    

class sigmoid(function): # Loss function
    # This function normalises the weights at the end, can be thought of as returning probabilities
    def __init__(self):
        self.type = 'normalisation'

    def forward(self,x,target):
        # Pass data through sigmoid function and normalise
        unnormalised_prob = np.exp(x-np.max(x,axis=1,keepdims=True))
        self.prob         = unnormalised_prob/np.sum(unnormalised_prob,axis=1,keepdims=True)
        self.target        = target
        # The loss is the log of the (exponentiated) measure, preserving its structure
        loss               = -np.log(self.prob[range(len(target)),target])
        return loss.mean() # The mean across the 1st dimension (i.e., the mean for each class across the samples)

    def backward(self):
        gradient = self.prob
        # If proba for a given target is 1 (correctly), then gradient is 0
        gradient[range(len(self.target)),self.target] -= 1.0 
        # Normalise the size of the gradients by the number of targets
        gradient/=len(self.target)
        return gradient

class ReLU(function): # Activation function
    def __init__(self,inplace=True):
        self.type = 'activation'
    # Simply pass forwards the positive values
    def forward(self,x):
        self.activated = x*(x>0)
        return self.activated
    
    def backward(self,d_y):
        return d_y*(self.activated>0) # Return the gradients without updating weights


class SGD(optimiser):
    # Stochastic gradient descent
    def __init__(self,parameters,lr=0.001,weight_decay=0.0,momentum=0.9):
        super().__init__(parameters) #Add to the default init
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.momentum       = momentum
        self.velocity       = []
        for p in parameters:
            self.velocity.append(np.zeros_like(p.grad))
            # Include momentum
        
    def step(self):
        for p,v in zip(self.parameters,self.velocity):
            # A bit more sophisticated than classical SGD, less so than Adam
            v = self.momentum*v + p.grad + self.weight_decay*p.data # 
            p.data = p.data-self.lr*v

class Model(): # Encapsulates a lot of the task of creating a model into a single task.  Similar to Keras
    def __init__(self):
        self.computation_graph  = []
        self.parameters         = []

    def add(self,layer):
        self.computation_graph.append(layer) # Appending the layers so we can loop through them
        self.parameters+=layer.get_params() # Appending the parameters to this list

    def __initialiseNetwork(self): # Initialiser
        for f in self.computation_graph:
            if f.type == 'linear':
                weights,bias = f.get_params()
                # initialise with shrunken samples from a random distribution
                weights.data = 0.01*np.random.randn(weights.size[0],weights.size[1])
                bias.data = 0.0
    def fit(self,data,target,batch_size,num_epochs,optimiser,loss_fn,data_gen,print_int = 500):
        # Time to run the model
        loss_history = []
        self.__initialiseNetwork()
        # Get our data
        data_gen = data_gen(data,target,batch_size)
        itr = 0
        for epoch in range(num_epochs):
            for X,Y in data_gen:
                # Start each epoch with zero gradients
                optimiser.zero_grad()
                # Complete a full pass through the network with the current sample
                for f in self.computation_graph:
                    X=f.forward(X)
                # Calculate the loss at the end of the pass
                loss = loss_fn.forward(X,Y)
                # Calculate the gradient based off this loss function
                grad = loss_fn.backward()
                for f in self.computation_graph[::-1]:
                    # Propogate the gradient backwards
                    grad = f.backward(grad)
                # Store loss for printing
                loss_history+= [loss]
                if np.mod(itr,print_int) == 0:
                    print("Epoch = {}, iteration = {}, loss = {}".format(epoch,itr,loss_history[-1]))
                itr += 1
                # Now gradients have been calculated for all the parameters, step through and update weights
                optimiser.step()

    def predict(self,data):
        X = data
        # Given data complete a forwards pass to classify the data
        for f in self.computation_graph:
            X = f.forward(X)
        return X
    