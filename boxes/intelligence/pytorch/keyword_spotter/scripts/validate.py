import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ptflops import get_model_complexity_info

# Locals libs
import dataset
import model

# Reimport
import importlib
importlib.reload(dataset)
importlib.reload(model)

# Get user name
username = os.getlogin()

# Specify paths
# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
project_path = repo_path + '/boxes/intelligence/pytorch/keyword_spotter'
dataset_path = project_path + '/_tmp/dataset'
output_path = project_path + '/_tmp'

# Prepare datasets
train_data, test_data, noise_data = dataset.prepare(dataset_path, 0.8)
target_distribution = np.histogram(train_data[1], bins=range(0,len(dataset.classes)+1))[0]
print(target_distribution)

# Create datasets
train_dataset = dataset.custom(wav_paths=train_data[0], targets=train_data[1], noise=noise_data, augment=True)
test_dataset = dataset.custom(wav_paths=test_data[0], targets=test_data[1], noise=noise_data, augment=False)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)

# Instantiate model
importlib.reload(model)
custom_model = model.custom()

# Reload saved model
model_path = model_path = project_path + '/_tmp/custom.pt'
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
macs, params = get_model_complexity_info(custom_model, (1, dataset.num_mfcc, dataset.num_times), as_strings=True, print_per_layer_stat=True, verbose=True)
print(f"MACs: {macs}")
print(f"Params: {params}")

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)
summary(custom_model, (1, dataset.num_mfcc, dataset.num_times))

# Compute validation here...

# Put model in eval mode
custom_model.eval()

# Store true and predicted labels
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        outputs = custom_model(inputs)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Get class labels (already defined in dataset.py as 'classes')
labels = dataset.classes

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix (Test Set)")
plt.savefig(f"{output_path}/confusion.png")

# FIN