import torch.nn as nn
import torch

"""
Neural network architectures for PINN model.

This module defines a collection of feedforward neural network architectures
that can be used within the proposed framework. The provided
models include:

- ANN         : standard fully connected feedforward neural network
- BranchedANN : shared-trunk network with multiple output heads
- Fourier_ANN : feedforward network with Fourier-feature augmentation

These architectures are designed to support flexible experimentation with
different network structures while maintaining a common PyTorch-based interface.
"""


class ANN(nn.Module):
    """
    Standard fully connected feedforward neural network.

    Inputs
    ------
    layer_size : list[int]
        List specifying the size of each layer, including input and output
        dimensions. For example, [dim_in, 64, 64, dim_out].

    activation : type[nn.Module]
        PyTorch activation class used after each hidden linear layer,
        e.g. `nn.Tanh`, `nn.ReLU`, or `nn.Sigmoid`.

    Returns
    -------
    torch.Tensor
        Predicted output tensor of shape `(batch_size, dim_out)`.

    Notes
    -----
    - All linear-layer weights are initialized using Xavier uniform
      initialization, and biases are initialized to zero.
    """

    def __init__(self, layer_size: list[int], activation: type[nn.Module]):
        super().__init__()

        layers = []

        for i in range(len(layer_size) - 2):
            layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            layers.append(activation())

        layers.append(nn.Linear(layer_size[-2], layer_size[-1]))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class SANN(nn.Module):
    """
    Standard fully connected feedforward neural network with softplus output activation.

    Inputs
    ------
    layer_size : list[int]
        List specifying the size of each layer, including input and output
        dimensions. For example, [dim_in, 64, 64, dim_out].

    activation : type[nn.Module]
        PyTorch activation class used after each hidden linear layer,
        e.g. `nn.Tanh`, `nn.ReLU`, or `nn.Sigmoid`.

    Returns
    -------
    torch.Tensor
        Predicted output tensor of shape `(batch_size, dim_out)`.

    Notes
    -----
    - All linear-layer weights are initialized using Xavier uniform
      initialization, and biases are initialized to zero.
    """

    def __init__(self, layer_size: list[int], activation: type[nn.Module]):
        super().__init__()

        layers = []

        for i in range(len(layer_size) - 2):
            layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            layers.append(activation())

        layers.append(nn.Linear(layer_size[-2], layer_size[-1]))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        raw_output = self.net(x)
        return self.softplus(raw_output)


class BranchedANN(nn.Module):
    """
    Branched feedforward neural network with a shared trunk and multiple heads.

    Inputs
    ------
    in_dim : int
        Input dimension of the network.

    trunk_layers : list[int]
        List specifying the hidden-layer sizes of the shared trunk network.

    head_dims : dict[str, int]
        Dictionary specifying the output heads, where each key is the head
        name and each value is the corresponding output dimension.

    activation : type[nn.Module]
        PyTorch activation class used after each hidden linear layer.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary mapping each head name to its predicted output tensor.

    Notes
    -----
    - The trunk learns a shared latent representation from the input.
    - Each output head receives the shared trunk representation and predicts
      a separate output block.
    - This architecture is useful when multiple outputs have related but
      distinct physical behavior.
    - All linear-layer weights are initialized using Xavier uniform
      initialization, and biases are initialized to zero.
    """

    def __init__(
        self,
        in_dim: int,
        trunk_layers: list[int],
        head_dims: dict[str, int],
        activation: type[nn.Module],
    ):
        super().__init__()

        trunk = []
        dims = [in_dim] + trunk_layers
        for i in range(len(dims) - 1):
            trunk.append(nn.Linear(dims[i], dims[i + 1]))
            trunk.append(activation())
        self.trunk = nn.Sequential(*trunk)

        hidden_dim = trunk_layers[-1]

        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, out_dim),
                )
                for name, out_dim in head_dims.items()
            }
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)
        outputs = {name: head(h) for name, head in self.heads.items()}
        return outputs


class Fourier_ANN(nn.Module):
    """
    Feedforward neural network with Fourier-feature augmentation.

    Inputs
    ------
    layer_size : list[int]
        List specifying the size of each layer before Fourier augmentation,
        including input and output dimensions.

    activation : type[nn.Module], optional, default=nn.Tanh
        PyTorch activation class used after each hidden linear layer.

    fourier_levels : int, optional, default=6
        Number of Fourier frequency levels used to augment the last input
        coordinate.

    Returns
    -------
    torch.Tensor
        Predicted output tensor of shape `(batch_size, dim_out)`.

    Notes
    -----
    - The last input coordinate is mapped to sinusoidal Fourier features
      using sine and cosine functions.
    - If `fourier_levels > 0`, the original last input variable is replaced
      by its Fourier-feature representation.
    - The augmented feature vector is then passed through a standard
      feedforward neural network.
    - A Softplus activation is applied to the final network output to enforce
      nonnegative predictions.
    """

    def __init__(
        self,
        layer_size: list[int],
        activation: type[nn.Module] = nn.Tanh,
        fourier_levels: int = 6,
        positive_output: bool = False,
    ):
        super().__init__()

        self.fourier_levels = fourier_levels
        self.positive_output = positive_output

        d_in = layer_size[0]
        net_layer_size = layer_size.copy()

        if self.fourier_levels > 0:
            extra = 2 * self.fourier_levels
            net_layer_size[0] = (d_in - 1) + extra

        layers = []
        for i in range(len(net_layer_size) - 2):
            layers.append(nn.Linear(net_layer_size[i], net_layer_size[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(net_layer_size[-2], net_layer_size[-1]))

        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.fourier_levels > 0:
            V = x[:, -1:].contiguous()
            x_base = x[:, :-1]

            device = x.device
            dtype = x.dtype

            omegas = (
                (2.0 ** torch.arange(self.fourier_levels, device=device, dtype=dtype))
                * torch.pi
            ).unsqueeze(0)

            args = V * omegas
            fourier_feats = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            x = torch.cat([x_base, fourier_feats], dim=1)

        raw = self.net(x)
        return self.softplus(raw) if self.positive_output else raw
