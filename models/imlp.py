from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from core.norms import alpha_norm


class IMLP(nn.Module, ABC):
    """ Implicit Multi Layer Perceptron """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 use_positional: bool,
                 positional_dim: int = 10,
                 skip_layers: tuple = (),
                 softmax: bool = False,
                 normalize: bool = False,
                 alpha_range: bool = False):

        super(IMLP, self).__init__()

        self.softmax = softmax
        self.normalize = normalize
        self.alpha_range = alpha_range

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional
        self.b = torch.Tensor()

        if self.use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim
            self.b = torch.tensor([(2 ** i) * np.pi for i in range(positional_dim)])
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers - 1:
                self.hidden.append(nn.Linear(input_dims, output_dim))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim))

    @staticmethod
    def positional_encoding_vec(x, b):
        proj = torch.einsum('ij, k -> ijk', x, b)
        mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)
        output = mapped_coords.transpose(2, 1).contiguous().flatten(1, -1)
        return output

    def forward(self, x):
        if self.use_positional:
            x = self.positional_encoding_vec(x, self.b)

        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)

            if i in self.skip_layers:
                x = torch.cat((x, input), 1)

            x = layer(x)

        if self.softmax:
            x = torch.softmax(x, dim=1)
        else:
            x = torch.tanh(x)

        if self.normalize:
            x = (x + 1.0) * 0.5

        if self.alpha_range:
            x = alpha_norm(x)

        return x

    def to(self, device: torch.device):
        if self.use_positional:
            self.b = self.b.to(device)

        return super(IMLP, self).to(device)


class StackedIMLP(nn.Module, ABC):
    def __init__(self, n_models: int = 1, *args, **kwargs):
        super(StackedIMLP, self).__init__()

        self.models = nn.ModuleList([IMLP(*args, **kwargs) for _ in range(n_models)])

    def forward(self, input: torch.Tensor):
        outputs = []

        for model in self.models:
            output = model(input)
            outputs.append(output)

        return torch.stack(outputs)
