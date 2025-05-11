import torch.nn as nn

class custom(nn.Module):
    def __init__(self, num_classes=12):
        super(custom, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1960, 256),   # 40 * 49  = 1960
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
