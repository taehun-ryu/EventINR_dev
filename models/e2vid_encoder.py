"""E2VID's full-resolution pre-prediction feature backbone (random initialization).

Adapted from uzh-rpg/rpg_e2vid, revision d0a7c005f460f2422f2a4bf605f70820ea7a1e5f,
model/unet.py and model/submodules.py (GPL-3.0; see LICENSE_E2VID.txt).
Matches the released E2VID configuration: three ConvLSTM levels, two residual
blocks, BatchNorm and sum skips. Only the prediction layer/sigmoid is omitted.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _ConvLayer(nn.Module):
    def __init__(self, inputs, outputs, stride=1, norm=True, kernel=5, activation=True):
        super().__init__()
        self.conv2d = nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2, bias=not norm)
        self.activation = activation
        if norm:
            self.norm_layer = nn.BatchNorm2d(outputs)

    def forward(self, x):
        x = self.conv2d(x)
        if hasattr(self, 'norm_layer'):
            x = self.norm_layer(x)
        return torch.relu(x) if self.activation else x


class _UpsampleConvLayer(_ConvLayer):
    def forward(self, x):
        return super().forward(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))


class _ConvLSTM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Gates = nn.Conv2d(2 * channels, 4 * channels, 3, padding=1)

    def forward(self, x, state=None):
        if state is None:
            state = (x.new_zeros(x.shape), x.new_zeros(x.shape))

        hidden, cell = state
        in_gate, remember_gate, out_gate, cell_gate = self.Gates(torch.cat((x, hidden), 1)).chunk(4, 1)
        cell = remember_gate.sigmoid() * cell + in_gate.sigmoid() * cell_gate.tanh()
        return out_gate.sigmoid() * cell.tanh(), cell


class _RecurrentConvLayer(nn.Module):
    def __init__(self, inputs, outputs, norm=True):
        super().__init__()
        self.conv = _ConvLayer(inputs, outputs, stride=2, norm=norm)
        self.recurrent_block = _ConvLSTM(outputs)

    def forward(self, x, state):
        state = self.recurrent_block(self.conv(x), state)
        return state[0], state


class _ResidualBlock(nn.Module):
    def __init__(self, channels, norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not norm)
        if norm:
            self.bn1, self.bn2 = nn.BatchNorm2d(channels), nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not norm)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out) if hasattr(self, 'bn1') else out)
        out = self.conv2(out)
        out = self.bn2(out) if hasattr(self, 'bn2') else out
        out += x
        return self.relu(out)


class E2VIDEncoder(nn.Module):
    """One voxel [B,bins,H,W] -> feature [B,channels,H,W] and three LSTM states.

    Includes the U-Net upsampling path. Padding/cropping follows EVREAL's
    centered zero-padding convention; recurrent states keep their padded size.
    Use bins=5, channels=32 for the released E2VID backbone's exact dimensions.
    """

    def __init__(self, bins=5, channels=32):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (bins, channels)):
            raise ValueError('bins and channels must be positive integers.')
        self.bins = bins
        widths = [channels * 2 ** i for i in range(4)]
        self.head = _ConvLayer(bins, channels, norm=False)
        self.encoders = nn.ModuleList(_RecurrentConvLayer(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.resblocks = nn.ModuleList(_ResidualBlock(widths[-1]) for _ in range(2))
        self.decoders = nn.ModuleList(_UpsampleConvLayer(c, c // 2) for c in reversed(widths[1:]))

    def forward(self, voxel, states=None):
        if states is None:
            states = (None,) * 3

        height, width = voxel.shape[-2:]
        pad_h, pad_w = (-height) % 8, (-width) % 8
        top, left = (pad_h + 1) // 2, (pad_w + 1) // 2
        x = self.head(F.pad(voxel, (left, pad_w // 2, top, pad_h // 2)))
        head, blocks, new_states = x, [], []
        for encoder, state in zip(self.encoders, states):
            x, state = encoder(x, state)
            blocks.append(x)
            new_states.append(state)

        for block in self.resblocks:
            x = block(x)
        for decoder, skip in zip(self.decoders, reversed(blocks)):
            x = decoder(x + skip)
        return (x + head)[..., top:top + height, left:left + width], tuple(new_states)
