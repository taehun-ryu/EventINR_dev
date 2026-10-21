"""E2VID, E2VID+ and ET-Net native 5-bin -> 32-channel pre-prediction graphs.

ET-Net / EVREAL 398f03551b7f5d150ae10aa89e4e731557fce240 model/eitr/u_trans.py;
E2VID and shared blocks adapted from uzh-rpg/rpg_e2vid revision
d0a7c005f460f2422f2a4bf605f70820ea7a1e5f model/unet.py and
model/submodules.py (GPL-3.0). E2VID+ follows TimoStoff/event_cnn_minimal
as preserved by the same EVREAL revision, with no normalization.
See vendor/LICENSE_SOURCES.md and the reproduced license notices.
Only the image prediction head is omitted. States are explicit and never detached.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .eitr.transformer_decoder import transformer_decoder
    from .eitr.transformer_encoder import transformer_encoder
else:
    from eitr.transformer_decoder import transformer_decoder
    from eitr.transformer_encoder import transformer_encoder


class _ConvLayer(nn.Module):
    def __init__(self, inputs, outputs, stride=1, norm=False):
        super().__init__()
        self.conv2d = nn.Conv2d(inputs, outputs, 5, stride, 2, bias=not norm)
        if norm:
            self.norm_layer = nn.BatchNorm2d(outputs)

    def forward(self, x):
        x = self.conv2d(x)
        if hasattr(self, 'norm_layer'):
            x = self.norm_layer(x)
        return torch.relu(x)


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
    def __init__(self, inputs, outputs, norm=False):
        super().__init__()
        self.conv = _ConvLayer(inputs, outputs, stride=2, norm=norm)
        self.recurrent_block = _ConvLSTM(outputs)

    def forward(self, x, state):
        state = self.recurrent_block(self.conv(x), state)
        return state[0], state


class _ResidualBlock(nn.Module):
    def __init__(self, channels, norm):
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
    """Released BatchNorm E2VID U-Net, omitting only the prediction head.

    One [B,5,H,W] voxel returns a [B,32,H,W] feature and three ConvLSTM
    states. Native widths are 32/64/128/256 with two residual blocks and sum
    skips. Centered padding/cropping follows EVREAL; states retain padding.
    """

    def __init__(self, *, _norm=True):
        super().__init__()
        widths = [32, 64, 128, 256]
        self.head = _ConvLayer(5, 32)
        self.encoders = nn.ModuleList(
            _RecurrentConvLayer(a, b, norm=_norm) for a, b in zip(widths[:-1], widths[1:]))
        self.resblocks = nn.ModuleList(_ResidualBlock(256, norm=_norm) for _ in range(2))
        self.decoders = nn.ModuleList(
            _UpsampleConvLayer(c, c // 2, norm=_norm) for c in reversed(widths[1:]))

    def forward(self, voxel, states=None):
        if states is None:
            states = (None,) * 3
        height, width = voxel.shape[-2:]
        pad_h, pad_w = (-height) % 8, (-width) % 8
        top, left = (pad_h + 1) // 2, (pad_w + 1) // 2
        x = self.head(F.pad(voxel, (left, pad_w // 2, top, pad_h // 2)))
        head, blocks, new_states = x, [], []
        for layer, state in zip(self.encoders, states):
            x, state = layer(x, state)
            blocks.append(x)
            new_states.append(state)
        for block in self.resblocks:
            x = block(x)
        for layer, skip in zip(self.decoders, reversed(blocks)):
            x = layer(x + skip)
        feature = (x + head)[..., top:top + height, left:left + width]
        return feature, tuple(new_states)


class E2VIDPlusEncoder(E2VIDEncoder):
    """Released E2VID+ U-Net: E2VID's native widths without normalization.

    Its published improvement also depends on sim-to-real training; choosing
    this graph does not load pretrained weights. States remain differentiable.
    """

    def __init__(self):
        super().__init__(_norm=False)


class _PositionEncoding(nn.Module):
    """Published flattened-token sine table (20,000 positions), vectorized."""

    def __init__(self):
        super().__init__()
        positions = np.arange(20000, dtype=np.float64)[:, None]
        indices = np.arange(256)[None, :]
        table = positions / np.power(10000, 2 * (indices // 2) / 256)
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        self.register_buffer('pos_table', torch.from_numpy(table).float()[None], persistent=False)

    def forward(self, words):
        if words.shape[1] > self.pos_table.shape[1]:
            raise ValueError('ET-Net supports at most 20,000 H/8 x W/8 tokens.')
        return self.pos_table[:, :words.shape[1]].to(device=words.device, dtype=words.dtype)


class ETNetEncoder(nn.Module):
    """One [B,5,H,W] voxel -> [B,32,H,W] feature and three ConvLSTM states.

    This preserves the released 32/64/128/256 widths, three 3-layer transformer
    encoders, three 2-layer decoders, eight attention heads, and six-way fusion.
    Global attention has quadratic memory cost in ceil(H/8)*ceil(W/8) tokens.
    """

    def __init__(self):
        super().__init__()
        self.head = _ConvLayer(5, 32)
        self.DownsampleConv = nn.ModuleList([
            _RecurrentConvLayer(32, 64),
            _RecurrentConvLayer(64, 128),
            _RecurrentConvLayer(128, 256),
        ])
        self.position_embedding = _PositionEncoding()
        self.split0 = nn.Unfold(kernel_size=1, stride=1, padding=0)
        for scale in range(3):
            if scale:
                setattr(self, f'split{scale}', nn.Conv2d(256 // 2 ** scale, 256,
                                                       2 ** scale, stride=2 ** scale))
            setattr(self, f'trans_encoder{scale}', transformer_encoder(
                d_model=256, nhead=8, num_encoder_layers=3,
                dim_feedforward=1024, activation='relu', dropout=0.1))
            setattr(self, f'trans_decoder{scale}', transformer_decoder(
                d_model=256, nhead=8, num_decoder_layers=2,
                dim_feedforward=1024, activation='relu', dropout=0.1))
        self.UpsampleConv = nn.ModuleList([
            _UpsampleConvLayer(256, 128),
            _UpsampleConvLayer(128, 64),
            _UpsampleConvLayer(64, 32),
        ])

    def forward(self, voxel, states=None):
        if states is None:
            states = (None,) * 3
        height, width = voxel.shape[-2:]
        pad_h, pad_w = (-height) % 8, (-width) % 8
        top, left = (pad_h + 1) // 2, (pad_w + 1) // 2
        x = self.head(F.pad(voxel, (left, pad_w // 2, top, pad_h // 2)))
        head, blocks, new_states = x, [], []
        for layer, state in zip(self.DownsampleConv, states):
            x, state = layer(x, state)
            blocks.append(x)
            new_states.append(state)
        encoded = []
        for scale in range(3):
            words = getattr(self, f'split{scale}')(blocks[-scale - 1]).flatten(2).transpose(1, 2)
            pos = self.position_embedding(words)
            encoded.append(getattr(self, f'trans_encoder{scale}')(
                src=words.transpose(0, 1), pos=pos.transpose(0, 1)))
        hs0, hs1, hs2 = encoded
        hc0 = self.trans_decoder0(tgt=hs0, memory=hs0)
        hc1 = self.trans_decoder1(tgt=hs1, memory=hs0)
        hc2 = self.trans_decoder2(tgt=hs2, memory=hs1)
        fused = (hs0 + hs1 + hs2 + hc0 + hc1 + hc2) / 6
        x = fused.permute(1, 2, 0).reshape(voxel.shape[0], 256, head.shape[-2] // 8, head.shape[-1] // 8)
        for layer, skip in zip(self.UpsampleConv, reversed(blocks)):
            x = layer(x + skip)
        feature = (x + head)[..., top:top + height, left:left + width]
        return feature, tuple(new_states)
