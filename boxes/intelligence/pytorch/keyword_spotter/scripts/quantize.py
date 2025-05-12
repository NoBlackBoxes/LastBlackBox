import os
import torch
import torch.nn as nn
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
output_folder = project_folder + '/_tmp/quantized'

# Create output folder (if it does not exist)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Prepare datasets
train_data, test_data, noise_data = dataset.prepare(dataset_folder, 0.8)

# Create datasets
train_dataset = dataset.custom(wav_paths=train_data[0], targets=train_data[1], noise=noise_data, augment=True)
test_dataset = dataset.custom(wav_paths=test_data[0], targets=test_data[1], noise=noise_data, augment=False)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)

# Instantiate model
custom_model = model.custom()

# Reload saved model
model_path = model_path = project_folder + '/_tmp/training/interim.pt'
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
custom_model.eval()

# Fuse modules
custom_model.fuse_model()

# Prepare for quantization
torch.backends.quantized.engine = "qnnpack"
custom_model.qconfig = get_default_qconfig("qnnpack")
prepared_model = prepare(custom_model)

# ---- Calibration ----
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = prepared_model(inputs)

# Convert to quantized model
quantized_model = convert(prepared_model)

# Save quantized model (TorchScript)
scripted_model = torch.jit.script(quantized_model)
scripted_model.save(f"{output_folder}/quantized.pt")

# ---- Optional: inspect layers ----
for name, module in quantized_model.named_modules():
    if isinstance(module, nn.quantized.Conv2d) or isinstance(module, nn.quantized.Linear):
        print(f"{name} → {module.__class__.__name__}")
        print(f"  Weight scale: {module.weight().q_scale()}")
        print(f"  Zero point: {module.weight().q_zero_point()}")

# FIN