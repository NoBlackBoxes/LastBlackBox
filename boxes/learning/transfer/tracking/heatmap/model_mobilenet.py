import torch


# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Load backbone
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_large', weights='MobileNet_V3_Large_Weights.DEFAULT')

        # Freeze the backbone weights
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze last layers
        unfrozen = [15,16]
        for index in unfrozen:
            for param in backbone.features[index].parameters():
                param.requires_grad = True

        # Remove classifier (i.e. extract feature detection layers)
        self.features =  torch.nn.Sequential(*list(backbone.children())[:-2])

        # Add a new prediction head
        self.conv = torch.nn.Conv2d(960, 1, kernel_size = (1,1))
        self.sigmoid = torch.nn.Sigmoid()

    # Forward
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
