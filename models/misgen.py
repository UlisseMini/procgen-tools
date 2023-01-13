"""
Interpretable versions of models from the goal misgeneralization paper.

Source from monte
https://gist.github.com/montemac/6ccf47f1e15349d82cff98f0ff5f30b1
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models.utils import orthogonal_init, xavier_uniform_init
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class MlpModel(nn.Module):
    def __init__(self,
                 input_dims=4,
                 hidden_dims=[64, 64],
                 **kwargs):
        """
        input_dim:     (int)  number of the input dimensions
        hidden_dims:   (list) list of the dimensions for the hidden layers
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(MlpModel, self).__init__()

        # Hidden layers
        hidden_dims = [input_dims] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        self.apply(orthogonal_init)

    def forward(self, x):
        for layer in self.layers:
           x = layer(x)


class NatureModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(NatureModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=64*7*7, out_features=512), nn.ReLU()
        )
        self.output_dim = 512
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.layers(x)
        return x

class ResidualAdd(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x1, x2):
    return x1 + x2

class InterpretableResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.resadd = ResidualAdd()

    def forward(self, x):
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.resadd(out, x)
        return out

class InterpretableImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = InterpretableResidualBlock(out_channels)
        self.res2 = InterpretableResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

# NOTE: scale varies between lauro and master branch, since we use both
# I monkeypatch this in model loading. Please, god, do not remove this constant.
scale = 1
class InterpretableImpalaModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        super().__init__()
        self.block1 = InterpretableImpalaBlock(in_channels=in_channels, out_channels=16*scale)
        self.block2 = InterpretableImpalaBlock(in_channels=16*scale, out_channels=32*scale)
        self.block3 = InterpretableImpalaBlock(in_channels=32*scale, out_channels=32*scale)
        self.relu3 = nn.ReLU()
        self.flatten = Flatten()
        self.reluflatten = nn.ReLU()
        self.fc = nn.Linear(in_features=32*scale * 8 * 8, out_features=256)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.reluflatten(x)
        return x





