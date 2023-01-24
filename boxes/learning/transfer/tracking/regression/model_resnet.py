import torch

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Load backbone
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')

        # Freeze the backbone weights
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze last layers
        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Remove classifier (i.e. extract feature detection layers)
        self.features =  torch.nn.Sequential(*list(backbone.children())[:-3])

        # Add a new prediction head
        self.conv = torch.nn.Conv2d(1024, 128, kernel_size = (1,1))
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(128*14*14,128)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(128,64)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.linear3 = torch.nn.Linear(64,32)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.dropout3 = torch.nn.Dropout(0.2)
        self.linear4 = torch.nn.Linear(32,2)
        self.sigmoid = torch.nn.Sigmoid()

    # Forward
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
