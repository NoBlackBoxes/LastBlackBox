import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class custom(nn.Module):
    def __init__(self, num_classes=12):
        super(custom, self).__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.conv_block1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 4, 5)),  # (left, right, top, bottom)
            nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.ds_conv_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.ds_conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.ds_conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.ds_conv_block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.ds_conv_block5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.conv_block1(x)
        x = self.ds_conv_block1(x)
        x = self.ds_conv_block2(x)
        x = self.ds_conv_block3(x)
        x = self.ds_conv_block4(x)
        x = self.ds_conv_block5(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
