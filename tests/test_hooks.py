import torch

from top_channels import LAYERS, zero_except


def test_zero_except_keeps_only_listed_channels_and_unhooks_cleanly():
    layer = torch.nn.Identity()
    x = torch.rand(2, 3, 4, 4) + 1
    handle = layer.register_forward_hook(zero_except([1]))
    out = layer(x)
    assert torch.equal(out[:, 1], x[:, 1])
    assert out[:, [0, 2]].eq(0).all()
    handle.remove()  # what featviz_dropout.py does before the "intact" measurement
    assert torch.equal(layer(x), x)


def test_zero_except_with_empty_keep_zeroes_everything():
    layer = torch.nn.Identity()
    layer.register_forward_hook(zero_except([]))
    assert layer(torch.ones(1, 3, 2, 2)).eq(0).all()


def test_previous_layer_of_first_block_is_last_conv():
    assert LAYERS[LAYERS.index("mixed3a") - 1] == "conv2d2"
