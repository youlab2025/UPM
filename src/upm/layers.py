"""Residual MLPs with parameter names compatible with DataV4L10 weights."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer, *, bias=True):
        super().__init__()
        if num_layer < 2:
            raise ValueError("MLPs require at least two layers")
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim, bias=bias)]
            + [nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(num_layer - 2)]
            + [nn.Linear(hidden_dim, out_dim, bias=bias)]
        )
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim, elementwise_affine=bias) for _ in range(num_layer - 1)]
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            if index == 0:
                x = self.layernorms[index](self.activation(layer(x)))
            elif index == len(self.layers) - 1:
                x = layer(x)
            else:
                x = self.layernorms[index](x + self.activation(layer(x)))
        return x


class MLPBiasFree(MLP):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer):
        super().__init__(in_dim, out_dim, hidden_dim, num_layer, bias=False)
