import math

import torch
import torch.nn as nn

import pytest

from pinnse import ANN, SANN, Fourier_ANN, MultiHeadANN, RecurrentANN


def linear_layers(module):
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def assert_xavier_init(module):
    for lin in linear_layers(module):
        bound = math.sqrt(6.0 / (lin.in_features + lin.out_features))
        assert torch.all(lin.weight.abs() <= bound + 1e-7)
        assert torch.count_nonzero(lin.bias) == 0


# ------------------------------------ ANN ------------------------------------
def test_ann_architecture_and_output_shape():
    model = ANN([3, 16, 8, 2], activation=nn.Tanh)

    lins = linear_layers(model)
    assert [(lin.in_features, lin.out_features) for lin in lins] == [
        (3, 16),
        (16, 8),
        (8, 2),
    ]
    # one activation after every hidden layer and none after the output layer
    assert [type(m) for m in model.net] == [nn.Linear, nn.Tanh] * 2 + [nn.Linear]
    assert model(torch.randn(5, 3)).shape == (5, 2)
    assert_xavier_init(model)


def test_ann_without_hidden_layers_is_affine():
    model = ANN([4, 2], activation=nn.ReLU)
    assert len(model.net) == 1
    x = torch.randn(6, 4)
    lin = model.net[0]
    torch.testing.assert_close(model(x), x @ lin.weight.T + lin.bias)


def test_ann_output_can_be_negative():
    model = ANN([1, 8, 1], activation=nn.Tanh)
    with torch.no_grad():
        model.net[-1].bias.fill_(-10.0)
    assert torch.all(model(torch.randn(4, 1)) < 0)


# ----------------------------------- SANN ------------------------------------
def test_sann_is_softplus_of_linear_stack():
    model = SANN([3, 8, 2], activation=nn.Tanh)
    with torch.no_grad():
        model.net[-1].bias.fill_(-10.0)

    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 2)
    assert torch.all(y > 0)
    torch.testing.assert_close(y, nn.functional.softplus(model.net(x)))
    assert_xavier_init(model.net[0])


# ------------------------------- MultiHeadANN --------------------------------
def test_multihead_returns_one_output_per_head():
    model = MultiHeadANN(
        in_dim=3,
        trunk_layers=[16, 8],
        head_dims={"flow": 4, "temp": 1},
        activation=nn.Tanh,
    )
    out = model(torch.randn(7, 3))

    assert set(out) == {"flow", "temp"}
    assert out["flow"].shape == (7, 4)
    assert out["temp"].shape == (7, 1)
    # each head starts from the last trunk width
    assert model.heads["flow"][0].in_features == 8
    assert_xavier_init(model)


def test_multihead_trunk_is_shared_and_heads_are_independent():
    model = MultiHeadANN(
        in_dim=2, trunk_layers=[8], head_dims={"a": 1, "b": 1}, activation=nn.Tanh
    )
    model(torch.randn(5, 2))["a"].sum().backward()

    assert model.trunk[0].weight.grad is not None
    assert model.heads["a"][0].weight.grad is not None
    assert model.heads["b"][0].weight.grad is None


# -------------------------------- Fourier_ANN --------------------------------
def test_fourier_input_width_is_expanded():
    model = Fourier_ANN([3, 16, 2], activation=nn.Tanh, fourier_levels=4)
    # two untouched inputs + sin/cos at four frequencies of the last input
    assert linear_layers(model)[0].in_features == 2 + 2 * 4
    assert model(torch.randn(5, 3)).shape == (5, 2)
    assert_xavier_init(model)


def test_fourier_features_match_manual_encoding():
    levels = 3
    model = Fourier_ANN([2, 4, 1], fourier_levels=levels)
    x = torch.tensor([[0.3, 0.25], [-0.7, -0.5]])

    v = x[:, -1:]
    omegas = torch.tensor([1.0, 2.0, 4.0]) * torch.pi
    feats = torch.cat([x[:, :1], torch.sin(v * omegas), torch.cos(v * omegas)], dim=1)

    torch.testing.assert_close(model(x), model.net(feats))


def test_fourier_features_are_periodic_in_last_input():
    model = Fourier_ANN([2, 8, 2], fourier_levels=5)
    x = torch.randn(6, 2)
    shifted = x.clone()
    shifted[:, -1] += 2.0  # lowest frequency is pi, so the period is 2

    torch.testing.assert_close(model(x), model(shifted), atol=1e-5, rtol=1e-5)


def test_fourier_with_zero_levels_is_plain_network():
    model = Fourier_ANN([3, 8, 2], fourier_levels=0)
    assert linear_layers(model)[0].in_features == 3
    x = torch.randn(4, 3)
    torch.testing.assert_close(model(x), model.net(x))


def test_fourier_positive_output_flag():
    raw = Fourier_ANN([2, 8, 1], fourier_levels=2, positive_output=False)
    pos = Fourier_ANN([2, 8, 1], fourier_levels=2, positive_output=True)
    pos.load_state_dict(raw.state_dict())
    with torch.no_grad():
        raw.net[-1].bias.fill_(-10.0)
        pos.net[-1].bias.fill_(-10.0)

    x = torch.randn(5, 2)
    assert torch.all(raw(x) < 0)
    assert torch.all(pos(x) > 0)
    torch.testing.assert_close(pos(x), nn.functional.softplus(raw(x)))


# ------------------------------- RecurrentANN --------------------------------
@pytest.mark.parametrize(
    "cell, cls",
    [("rnn", nn.RNN), ("gru", nn.GRU), ("lstm", nn.LSTM), ("LSTM", nn.LSTM)],
)
def test_recurrent_cells_and_output_shape(cell, cls):
    model = RecurrentANN(in_dim=3, hidden_dim=8, out_dim=2, n_layers=2, cell=cell)
    assert isinstance(model.recurrent, cls)
    assert model.cell == cell.lower()
    assert model.recurrent.num_layers == 2
    assert model(torch.randn(4, 10, 3)).shape == (4, 10, 2)


def test_recurrent_head_layers():
    model = RecurrentANN(
        in_dim=2, hidden_dim=8, out_dim=3, head_layers=[16, 4], activation=nn.ReLU
    )
    assert [type(m) for m in model.head] == [nn.Linear, nn.ReLU] * 2 + [nn.Linear]
    assert [
        (lin.in_features, lin.out_features) for lin in linear_layers(model.head)
    ] == [
        (8, 16),
        (16, 4),
        (4, 3),
    ]
    assert_xavier_init(model.head)


def test_recurrent_default_head_is_single_linear_layer():
    model = RecurrentANN(in_dim=2, hidden_dim=8, out_dim=3)
    assert len(model.head) == 1 and isinstance(model.head[0], nn.Linear)


def test_recurrent_is_causal():
    model = RecurrentANN(in_dim=2, hidden_dim=8, out_dim=1)
    x = torch.randn(3, 8, 2)
    perturbed = x.clone()
    perturbed[:, 5:] += 1.0

    y, y_p = model(x), model(perturbed)
    torch.testing.assert_close(y[:, :5], y_p[:, :5])
    assert not torch.allclose(y[:, 5:], y_p[:, 5:])


def test_recurrent_rejects_unknown_cell():
    with pytest.raises(ValueError, match="Unknown cell 'transformer'"):
        RecurrentANN(in_dim=2, hidden_dim=4, out_dim=1, cell="transformer")


def test_recurrent_rejects_non_sequence_input():
    model = RecurrentANN(in_dim=2, hidden_dim=4, out_dim=1)
    with pytest.raises(ValueError, match=r"\(batch, seq_len, in_dim\)"):
        model(torch.randn(5, 2))
