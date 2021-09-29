import numpy as np 
import sys
import fresh_dl as fdl
import data_funcs as dtf

"""
Example call - 
    python test_dl.py fig_title 0.2
"""

data_type = 'default'
if len(sys.argv) < 3:
    noise = 0.2
if len(sys.argv) < 2:
    fit_title = ''
if len(sys.argv) > 3:
    data_type = sys.argv[3]

fig_title = sys.argv[1]
noise     = float(sys.argv[2])
raw_name  = 'raw' + fig_title + '.png'
class_title='classified' + fig_title + '.png'

batch_size = 20
num_epochs = 200
samples_per_class = 100
num_classes = 3
hidden_units = 100
data,target = dtf.gen_spiral_data(samples_per_class,num_classes,noise,data_type)
raw_data = dtf.plot_scatter(data,target)
raw_data.figure.savefig(raw_name)
model = fdl.Model()
model.add(fdl.Linear(2,hidden_units))
model.add(fdl.ReLU())
model.add(fdl.Linear(hidden_units,num_classes))
optimiser = fdl.SGD(model.parameters,lr = 1,weight_decay = 0.001,momentum = 0.9)
loss  = fdl.sigmoid()
model.fit(data,target,batch_size,num_epochs,optimiser,loss,dtf.data_generator)
pre_arg = model.predict(data)
pred_labels = np.argmax(pre_arg,axis=1)
good_labels = pred_labels == target
accuracy    = np.sum(good_labels)/len(target)
print("Model Accuracy = {:.2f}%".format(accuracy*100))
classified_data = dtf.plot_decision(data,target,model)
classified_data.figure.savefig(class_title)
