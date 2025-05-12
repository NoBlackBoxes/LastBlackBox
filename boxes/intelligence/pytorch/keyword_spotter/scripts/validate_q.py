import os
import torch
import platform
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.quantization import prepare, convert, get_default_qconfig

# Locals libs
import dataset
#import model_dnn as model
#import model_cnn as model
import model_dscnn as model

# Reimport
import importlib
importlib.reload(dataset)
importlib.reload(model)

# Specify paths
username = os.getlogin()
LBB = '/home/' + username + '/NoBlackBoxes/LastBlackBox'
project_folder = LBB + '/boxes/intelligence/pytorch/keyword_spotter'
dataset_folder = project_folder + '/_tmp/dataset'
output_folder = project_folder + '/_tmp/validation'

# Create output folder (if it does not exist)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Detect machine architecture
arch = platform.machine().lower()

# Prepare datasets
train_data, test_data, noise_data = dataset.prepare(dataset_folder, 0.8)
target_distribution = np.histogram(train_data[1], bins=range(0,len(dataset.classes)+1))[0]

# Create datasets
train_dataset = dataset.custom(wav_paths=train_data[0], targets=train_data[1], noise=noise_data, augment=True)
test_dataset = dataset.custom(wav_paths=test_data[0], targets=test_data[1], noise=noise_data, augment=False)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)

# Select torch backend
if 'arm' in arch or 'aarch64' in arch:
    torch.backends.quantized.engine = 'qnnpack'
else:
    torch.backends.quantized.engine = 'fbgemm'
print("Quantized backend set to:", torch.backends.quantized.engine)

# Load model
model_path = project_folder + '/_tmp/quantized/quantized.pt'
quantized_model = torch.jit.load(model_path)
quantized_model.eval()
quantized_model.to("cpu")

# Store true and predicted labels
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to("cpu")
        outputs = quantized_model(inputs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute accuracy
y_true = np.array(y_true)
y_pred = np.array(y_pred)
correct = np.sum(y_true==y_pred)
accuracy = 100.0 * (correct/len(y_true))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Get class labels (already defined in dataset.py as 'classes')
labels = dataset.classes

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.suptitle("Confusion Matrix (Test Set) : Quantized Model")
plt.title(f"Accuracy: {accuracy:.2f}%")
plt.savefig(f"{output_folder}/confusion_quantized.png")

# FIN