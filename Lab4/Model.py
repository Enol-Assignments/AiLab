# model.py
import torch
import torch.nn as nn
import torchvision


class ResNetBaseline(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetBaseline, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)


class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CNNLSTMModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((64, 1))
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_time_step))
        out = self.fc2(out)
        return out
