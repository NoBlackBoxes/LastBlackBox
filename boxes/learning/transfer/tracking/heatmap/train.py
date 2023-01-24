import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchsummary import summary
import cv2

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
output_path = box_path + '/_tmp'

# Specify transforms for inputs
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prepare datasets
train_data, test_data = dataset.prepare('train2017', 0.8)

# Create datasets
train_dataset = dataset.custom(image_paths=train_data[0], targets=train_data[1], transform=preprocess, augment=True)
test_dataset = dataset.custom(image_paths=test_data[0], targets=test_data[1], transform=preprocess, augment=True)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# Inspect dataset?
inspect = True
if inspect:
    train_features, train_targets = next(iter(train_dataloader))
    for i in range(9):
        plt.subplot(3,3,i+1)
        feature = train_features[i]
        target = np.squeeze(train_targets[i].numpy())
        feature = (feature + 2.0) / 4.0
        image = np.transpose(feature, (1,2,0))
        heatmap = cv2.resize(target, (224,224))
        plt.imshow(image, alpha=0.75)
        plt.imshow(heatmap, alpha=0.5)
    plt.show()

# Instantiate model
importlib.reload(model)
custom_model = model.custom()

# Set loss function
loss_fn = torch.nn.MSELoss(reduction ="mean")
optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.001)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Move model to device
custom_model.to(device)
summary(custom_model, (3, 224, 224))

# Define training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = custom_model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            pixel_loss = np.sqrt(loss) * 224.0
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], pixel_loss: {pixel_loss:>5f}")

# Define testing
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    pixel_loss = np.sqrt(test_loss) * 224.0
    print(f"Test Error: \n Avg loss: {test_loss:>8f}, pixel_loss: {pixel_loss:>5f}\n")

# TRAIN
epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer)
    test(test_dataloader, custom_model, loss_fn)
print("Done!")




# Display image and label.
train_features, train_targets = next(iter(test_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Targets batch shape: {train_targets.size()}")

# Let's run it
train_features_gpu = train_features.to(device)
outputs = custom_model(train_features_gpu)
#outputs = outputs.cpu()
#loss_fn = torch.nn.MSELoss(reduction ="sum")
#loss_fn(outputs, train_targets)
outputs = outputs.cpu().detach().numpy()


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
#    plt.imshow(target_heatmap, alpha=0.5)
plt.show()




# Save model
torch.save(custom_model.state_dict(), output_path + '/custom.pt')


# FIN