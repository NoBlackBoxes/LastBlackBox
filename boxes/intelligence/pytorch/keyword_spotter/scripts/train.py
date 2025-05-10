import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
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

# Inspect dataset?
inspect = False
if inspect:
    for i in range(90):
        path = train_data[0][i]
        target = train_data[1][i]
        print(path, dataset.classes[target])
    train_features, train_targets = next(iter(train_dataloader))
    for i in range(9):
        plt.subplot(3,3,i+1)
        feature = np.squeeze(train_features[i])
        target = train_targets[i]
        plt.imshow(feature, alpha=0.75)
    plt.savefig("_tmp/inspect.png")
    plt.show()

# Instantiate model
importlib.reload(model)
custom_model = model.custom()
macs, params = get_model_complexity_info(custom_model, (1, dataset.num_mfcc, dataset.num_times), as_strings=True, print_per_layer_stat=True, verbose=True)
print(f"MACs: {macs}")
print(f"Params: {params}")

## Reload saved model
#model_path = model_path = box_path + '/_tmp/custom.pt'
#custom_model = model.custom()
#custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set optimizer
#optimizer = torch.optim.AdamW(custom_model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000, 8000], gamma=0.2)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Set loss function
total_samples = np.sum(target_distribution)
pos_weights = (total_samples - target_distribution) / target_distribution
#pos_weights = np.log1p(pos_weights)
pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
#loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
#loss_fn = torch.nn.BCELoss()
loss_fn = torch.nn.CrossEntropyLoss()

# Move model to device
custom_model.to(device)
summary(custom_model, (1, dataset.num_mfcc, dataset.num_times))

# Define accuracy
def measure_accuracy(targets, guesses):
    preds = guesses.argmax(dim=1)
    correct = (preds == targets).sum().item()
    wrong = targets.size(0) - correct
    return correct, wrong

# Define training
def train(_dataloader, _model, _loss_function, _optimizer, _scheduler):
    size = len(_dataloader.dataset)
    _model.train()
    for batch, (X, y) in enumerate(_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = _model(X)
        loss = _loss_function(pred, y)

        # Backpropagation
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()
        _scheduler.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            correct, wrong = measure_accuracy(y, pred)
            step_count = scheduler._step_count
            print(f"{step_count-2}: {correct} vs {wrong} : {100.0*correct/(correct+wrong):.2f}%, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define testing
def test(_dataloader, _model, _loss_function):
    size = len(_dataloader.dataset)
    num_batches = len(_dataloader)
    _model.eval()
    test_loss = 0.0
    accum_correct = 0
    accum_wrong = 0
    with torch.no_grad():
        for X, y in _dataloader:
            X, y = X.to(device), y.to(device)
            pred = _model(X)
            test_loss += _loss_function(pred, y).item()
            correct, wrong = measure_accuracy(y, pred)
            accum_correct = accum_correct + correct
            accum_wrong = accum_wrong + wrong
    avg_test_loss = test_loss / num_batches
    print(f"Test Results: {accum_correct} vs {accum_wrong} : {100.0 * accum_correct/(accum_correct+accum_wrong):.2f}%\n Avg loss: {avg_test_loss:>8f}\n")

# TRAIN
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, custom_model, loss_fn, optimizer, scheduler)
    test(test_dataloader, custom_model, loss_fn)

    # Save interim model
    torch.save(custom_model.state_dict(), output_path + '/custom.pt')
print("Done!")

# Save model
torch.save(custom_model.state_dict(), output_path + '/final.pt')

# Reload saved model
#model_path = model_path = box_path + '/_tmp/custom.pt'
#custom_model = model.custom()
#custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))



# FIN