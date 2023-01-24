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
        self.conv = torch.nn.Conv2d(1024, 1, kernel_size = (1,1))
        self.sigmoid = torch.nn.Sigmoid()

    # Forward
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
