import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from pycocotools.coco import COCO

# Locals libs
import model
import dataset

# Reimport
import importlib
importlib.reload(dataset)
importlib.reload(model)

# Specify paths
repo_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/ai/tracking/nose/heatmap'
model_path = box_path + '/_tmp/custom.pt'

# Specify transforms for inputs
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prepare datasets
train_data, test_data = dataset.prepare('train2017', 0.8)

# Create datasets
train_dataset = dataset.custom(image_paths=train_data[0], targets=train_data[1], transform=preprocess)
test_dataset = dataset.custom(image_paths=test_data[0], targets=test_data[1], transform=preprocess)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# Load model
custom_model = model.custom()
custom_model.load_state_dict(torch.load(model_path))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)

# Display image and label.
train_features, train_targets = next(iter(test_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Targets batch shape: {train_targets.size()}")

# Let's run it
train_features_gpu = train_features.to(device)
outputs = custom_model(train_features_gpu)
outputs = outputs.cpu().detach().numpy()

# Examine predictions
for i in range(9):
    plt.subplot(3,3,i+1)
    feature = train_features[i]
    target = np.squeeze(train_targets[i].numpy())
    feature = (feature + 2.0) / 4.0
    image = np.transpose(feature, (1,2,0))
    target_heatmap = cv2.resize(target, (224,224))
    output = np.squeeze(outputs[i])
    predicted_heatmap = cv2.resize(output, (224,224))
    plt.imshow(image, alpha=0.75)
    plt.imshow(predicted_heatmap, alpha=0.5)
plt.show()

# Specify transforms for inputs
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test on movie snaps
import cv2
video_path = repo_path + '/boxes/ai/tracking/_data/nose.mp4'
cap = cv2.VideoCapture(video_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(9):
    plt.subplot(3,3,i+1)

    ret = cap.set(cv2.CAP_PROP_POS_FRAMES, i * 20)
    ret, frame = cap.read()
    resized = cv2.resize(frame, (224,224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    feature = torch.unsqueeze(preprocess(rgb), 0)
    feature_gpu = feature.to(device)
    output = custom_model(feature_gpu)

    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    predicted_heatmap = cv2.resize(output, (224,224))

    plt.imshow(rgb, alpha=0.75)
    plt.imshow(predicted_heatmap, alpha=0.5)
plt.show()

cap.release()

#FIN