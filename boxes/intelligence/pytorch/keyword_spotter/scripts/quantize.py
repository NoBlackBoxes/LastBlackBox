import os
import torch
import torch.nn as nn
from torch.quantization import convert, get_default_qconfig

# Locals libs
#import model_dnn as model
#import model_cnn as model
import model_dscnn as model

# Reimport
import importlib
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

# Reload trained model
model_path = model_path = project_folder + '/_tmp/training/interim.pt'
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
custom_model.eval()

# Fuse modules
custom_model.fuse_model()

# Specify backend
torch.backends.quantized.engine = "qnnpack"
custom_model.qconfig = get_default_qconfig("qnnpack")

# Convert to quantized model
quantized_model = convert(custom_model)

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