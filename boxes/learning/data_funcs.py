import numpy as np
import matplotlib.pyplot as plt
import warnings

def gen_spiral_data(points_per_class,num_classes,noise = 0.2,data_type='default'):
    data = np.ndarray((points_per_class*num_classes,2),np.float32)
    target = np.ndarray((points_per_class*num_classes,),np.uint8)
    if data_type == 'default':
        r = np.linspace(0,1,points_per_class)
        radians_per_class = 2*np.pi/num_classes
        for i in range(num_classes):
            t = np.linspace(i*radians_per_class,(i+2.5)*radians_per_class,points_per_class)
            t += noise*np.random.randn(points_per_class) # Add in noise to the classification
            data[i*points_per_class:(i+1)*points_per_class] = np.c_[r*np.cos(t),2*r*np.sin(t)]
            target[i*points_per_class:(i+1)*points_per_class] = i

    elif data_type == 'double':
        r = np.linspace(0,1,points_per_class)
        radians_per_class = 2*np.pi/num_classes
        num_t1 = int(points_per_class/2)
        num_t2 = points_per_class - num_t1
        for i in range(num_classes):
            t1 = np.linspace(i*radians_per_class,(i+0.5)*radians_per_class,num_t1)
            t1 += noise*np.random.randn(num_t1) # Add in noise to the classification
            t2 = t1 + 5
            t  = np.concatenate((t1,t2))
            
            data[i*points_per_class:(i+1)*points_per_class] = np.c_[r*np.cos(t),2*r*np.sin(t)]
            target[i*points_per_class:(i+1)*points_per_class] = i
    ####################
    ## Add extra here ##
    ####################
    else:
        warnings.warn("Unrecognised data request, producing default data")
        r = np.linspace(0,1,points_per_class)
        radians_per_class = 2*np.pi/num_classes
        for i in range(num_classes):
            t = np.linspace(i*radians_per_class,(i+2.5)*radians_per_class,points_per_class)
            t += noise*np.random.randn(points_per_class) # Add in noise to the classification
            data[i*points_per_class:(i+1)*points_per_class] = np.c_[r*np.cos(t),2*r*np.sin(t)]
            target[i*points_per_class:(i+1)*points_per_class] = i
    return data,target


class data_generator():
    def __init__(self,data,target,batch_size,shuffle=True):
        self.shuffle = shuffle
        if shuffle:
            shuffled_ind = np.random.permutation(len(data))
        else:
            shuffled_ind = range(len(data))
        
        self.data = data[shuffled_ind]
        self.target = target[shuffled_ind]
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(data.shape[0]/batch_size))
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.num_batches:
            batch_data = self.data[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
            batch_target = self.target[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
            self.counter += 1
            return batch_data,batch_target
        else:
            if self.shuffle:
                shuffled_ind = np.random.permutation(len(self.target))
            else:
                shuffled_ind = range(len(self.target))

            self.data = self.data[shuffled_ind]
            self.target = self.target[shuffled_ind]
            self.counter = 0
            raise StopIteration

def plot_scatter(data,target):
    plt.scatter(x=data[:,0],y=data[:,1],c = target,cmap = plt.cm.Dark2)
    return plt.gca()

def plot_decision(data,target,model):
    x_min,x_max = np.min(data[:,0])-0.5,np.max(data[:,0])+0.5
    y_min,y_max = np.min(data[:,1])-0.5,np.max(data[:,1])+0.5

    x,y = np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01)
    xx,yy = np.meshgrid(x,y)

    z = np.argmax(model.predict(np.c_[xx.ravel(),yy.ravel()]),axis=1).reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap = plt.cm.tab10)
    plt.scatter(data[:,0],data[:,1],c=target,cmap = plt.cm.Accent)
    return plt.gca()