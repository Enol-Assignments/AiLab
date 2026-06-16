import torch
import torchvision
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = nn.Linear(512, 8)

    def forward(self, x):
        return self.resnet(x)