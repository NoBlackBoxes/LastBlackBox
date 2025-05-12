import torch.nn as nn
from torch.quantization import fuse_modules

class custom(nn.Module):
    def __init__(self, num_classes=12):
        super(custom, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(20, 8), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(10, 4), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # Controls final size for flatten

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
    def fuse_model(self):
        fuse_modules(self.conv_block1, [['0', '1', '2']], inplace=True)
        fuse_modules(self.conv_block2, [['0', '1', '2']], inplace=True)
        fuse_modules(self.conv_block3, [['0', '1', '2']], inplace=True)
        return
