import torch.nn as nn
import torch.nn.functional as F


class MLP_MNIST(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class SimpleCNN_CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)  # 32→16→8 と2回使う
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)   # (B,32,32,32)
        x = self.pool(x)    # (B,32,16,16)
        x = self.conv2(x)   # (B,64,16,16)
        x = self.pool(x)    # (B,64,8,8)
        x = self.conv3(x)   # (B,128,8,8)
        x = x.flatten(1)    # (B, 128*8*8)
        x = self.fc(x)
        return x 