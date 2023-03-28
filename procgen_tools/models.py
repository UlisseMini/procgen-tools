"""
Interpretable versions of models from the goal misgeneralization paper.

Source from monte
https://gist.github.com/montemac/6ccf47f1e15349d82cff98f0ff5f30b1
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from bidict import bidict
import torch
import numpy as np

# type ignores are because of bad/inconsistent typing on gain

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain) # type: ignore
        nn.init.constant_(module.bias.data, 0) # type: ignore
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0) # type: ignore
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1) # Skip batch dimension


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
        self.fc = nn.Linear(in_features=32*scale * 8 * 8, out_features=256)
        self.relufc = nn.ReLU()

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relufc(x)
        return x

class CategoricalPolicy(nn.Module):
    """
    Copied from train-procgen-pytorch, removed recurrent option as we're not using it.
    """
    def __init__(self, embedder, action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        # small scale weight-initialization in policy enhances the stability        
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

    def forward(self, x):
        hidden = self.embedder(x)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v


# TODO: We should probably move these to a separate file, this isn't model code.

# DO NOT CHANGE ORDERING IN DICTS. ORDERING MATTERS. FILES DEPEND ON IT.

MAZE_ACTION_INDICES = {
    'LEFT': [0, 1, 2],
    'DOWN': [3],
    'UP': [5],
    'RIGHT': [6, 7, 8],
    'NOOP': [4,9,10,11,12,13,14],
}

# action deltas. we index from bottom left by (row, col)
MAZE_ACTION_DELTAS = bidict({
    'LEFT': (0, -1),
    'RIGHT': (0, 1),
    'UP': (1, 0),
    'DOWN': (-1, 0),
    'NOOP': (0, 0),
})

# TODO: clean this up
MAZE_ACTIONS_BY_INDEX = np.zeros((15), dtype='<U5')
MAZE_ACTION_DELTAS_BY_INDEX = np.zeros((15,2))
for act, inds in MAZE_ACTION_INDICES.items():
    MAZE_ACTION_DELTAS_BY_INDEX[inds,:] = np.array(MAZE_ACTION_DELTAS[act])
    MAZE_ACTIONS_BY_INDEX[inds] = act


def human_readable_action(act: int) -> str:
    """
    Convert an action index to a human-readable action name.
    The original action space is 15 actions, but we only care about 5 of them in this maze environment.
    """
    assert act in range(15), f'{act} is not in range(15)'
    return next(act_name for act_name, act_indexes in MAZE_ACTION_INDICES.items() if act in act_indexes)


def human_readable_actions(probs: np.ndarray) -> dict:
    """
    Convert a categorical distribution to a human-readable dict of actions, with probabilities.
    The original action space is 15 actions, but we only care about 5 of them in this maze environment.
    """
    if isinstance(probs, Categorical): # backwards compat
        probs = probs.probs

    return {act_name: probs[..., np.array(act_indexes)].sum(-1) for act_name, act_indexes in MAZE_ACTION_INDICES.items()}


def load_policy(model_file: str, action_size: int, device = None) -> CategoricalPolicy:
    assert type(action_size) == int

    checkpoint = torch.load(model_file, map_location=device)

    # CURSED. scale varies between models trained on the lauro vs. master branch. 
    global scale
    scale = checkpoint['model_state_dict']['embedder.block1.conv.weight'].shape[0]//16

    model = InterpretableImpalaModel(in_channels=3)
    policy = CategoricalPolicy(model, action_size=action_size)
    policy.load_state_dict(checkpoint['model_state_dict'])
    return policy


def num_channels(hook, layer_name: str):
    """ Get the number of channels in the given layer. """
    # Ensure hook has been run on dummy input
    assert hook.get_value_by_label(layer_name) is not None, "Hook has not been run on any input"
    return hook.get_value_by_label(layer_name).shape[1]

