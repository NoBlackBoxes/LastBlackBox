import os
import torch
import ai_edge_torch

# Locals libs
import dataset
import model

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/OtherBlackBoxes'
box_path = repo_path + '/ai/speech/keyword'
output_path = box_path + '/_tmp'
dataset_folder = box_path + '/_tmp/dataset'

# Instantiate model
custom_model = model.custom()

# Reload saved model
model_path = model_path = box_path + '/_tmp/custom.pt'
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Convert to TF Lite model
sample_input = (torch.randn((1, 1, dataset.num_mfcc, dataset.num_times)),)
edge_model = ai_edge_torch.convert(custom_model.eval(), sample_input)
edge_model.export(box_path + '/_tmp/custom.tflite')

#FIN