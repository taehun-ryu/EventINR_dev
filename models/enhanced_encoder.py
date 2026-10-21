"""Faithful E2VID+ and ET-Net pre-prediction backbones for the XYT INR.

Both preserve the released 32/64/128/256 convolution widths. The original
image/flow prediction head is omitted; a 32 -> channels projection is added
only when channels != 32. Forward state is explicit and never detached.
See vendor/PROVENANCE.md for sources, exact adaptation boundaries and licenses.
"""

from collections import OrderedDict
from collections.abc import Mapping
import pickle
from pathlib import PurePosixPath

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .e2vid_encoder import (
    E2VIDEncoder, _ConvLayer, _RecurrentConvLayer, _ResidualBlock, _UpsampleConvLayer,)
from eitr.transformer_decoder import transformer_decoder
from eitr.transformer_encoder import transformer_encoder


class _ConfigMetadata:
    """Inert storage for legacy checkpoint metadata; never imports its code."""


class _RestrictedUnpickler(pickle.Unpickler):
    """PyTorch 2.1 compatibility: resolve only audited tensor/metadata globals.

    This is not the ordinary pickle loader: unknown names always raise, and
    external ConfigParser code is never imported or instantiated.
    """

    def find_class(self, module, name):
        allowed = {
            ('collections', 'OrderedDict'): OrderedDict,
            ('parse_config', 'ConfigParser'): _ConfigMetadata,
            ('pathlib', 'PosixPath'): PurePosixPath,
            ('torch', 'FloatStorage'): torch.FloatStorage,
            ('torch', 'LongStorage'): torch.LongStorage,
            ('torch._utils', '_rebuild_tensor_v2'): torch._utils._rebuild_tensor_v2,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Forbidden checkpoint global: {module}.{name}')
        return allowed[module, name]


class _RestrictedPickle:
    Unpickler = _RestrictedUnpickler

    @staticmethod
    def load(handle, **kwargs):
        return _RestrictedUnpickler(handle, **kwargs).load()


def _read_checkpoint(path):
    """Restricted tensor loading, including EVREAL's inert legacy metadata.

    PyTorch >=2.6 uses weights_only with two inert metadata allowances. On
    PyTorch 2.1, the explicit restricted pickle module resolves only the six
    audited globals above. There is no unrestricted deserialization path.
    """
    # Named aliases in safe_globals are supported from 2.6 (the API itself
    # existed earlier, but accepted only classes, not (class, old_name) pairs).
    torch_version = tuple(int(part) for part in torch.__version__.split('.')[:2])
    if torch_version >= (2, 6) and hasattr(torch.serialization, 'safe_globals'):
        allowed = [(_ConfigMetadata, 'parse_config.ConfigParser'),
                   (PurePosixPath, 'pathlib.PosixPath')]
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location='cpu', weights_only=True)
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError:
        # Legacy ConfigParser-bearing files predate the public safe_globals API.
        # The supplied module REPLACES pickle with the fail-closed allowlist.
        return torch.load(path, map_location='cpu', weights_only=False,
                          pickle_module=_RestrictedPickle)


class _ReleasedBackbone:
    native_channels = 32

    def _configure_output(self, bins, channels):
        if any(type(value) is not int or value < 1 for value in (bins, channels)):
            raise ValueError('bins and channels must be positive integers.')
        self.bins, self.out_channels = bins, channels
        self.projection = nn.Identity() if channels == 32 else nn.Conv2d(32, channels, 1)

    @property
    def metadata(self):
        projection = sum(p.numel() for p in self.projection.parameters())
        total = sum(p.numel() for p in self.parameters())
        return dict(name=self.backbone_name, native_channels=32,
                    out_channels=self.out_channels, bins=self.bins,
                    backbone_parameters=total - projection,
                    projection_parameters=projection, total_parameters=total,
                    recurrent_levels=3,
                    pretrained=getattr(self, '_initialization', 'random') != 'random',
                    initialization=getattr(self, '_initialization', 'random'),
                    prediction_head_omitted=True)

    def load_reconstruction_checkpoint(self, checkpoint):
        """Load *all* backbone tensors, omitting only reconstruction pred.*.

        Accept a checkpoint path, a {state_dict: ...} mapping or a plain tensor
        state_dict. EVREAL prefixes module./unetflow./unetrecurrent. are handled.
        Missing, unexpected, or shape-incompatible backbone tensors raise before
        any parameters are changed. A non-32-channel output projection remains
        randomly initialized and trainable. Returns a small loading report.
        """
        payload = checkpoint if isinstance(checkpoint, Mapping) else _read_checkpoint(checkpoint)
        state = payload.get('state_dict', payload) if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise ValueError('Expected a checkpoint containing a tensor state_dict.')
        converted, ignored = {}, []
        for original, value in state.items():
            if not isinstance(original, str) or not isinstance(value, torch.Tensor):
                raise ValueError('The state_dict must map string names to tensors.')
            key = original.removeprefix('module.')
            for prefix in ('unetrecurrent.', 'unetflow.'):
                key = key.removeprefix(prefix)
            if key.startswith('pred.'):
                ignored.append(original)
                continue
            if key in converted:
                raise ValueError(f'Duplicate normalized checkpoint key: {key}')
            converted[key] = value
        current = self.state_dict()
        expected = {key for key in current if not key.startswith('projection.')}
        missing, extra = sorted(expected - converted.keys()), sorted(converted.keys() - expected)
        mismatched = [key for key in expected & converted.keys()
                      if current[key].shape != converted[key].shape]
        if missing or extra or mismatched:
            raise ValueError(f'Incompatible {self.backbone_name} checkpoint: '
                             f'missing={missing}, unexpected={extra}, shape_mismatch={mismatched}')
        current.update(converted)
        self.load_state_dict(current, strict=True)
        self._initialization = 'reconstruction_checkpoint'
        return dict(loaded_tensors=len(converted), omitted_prediction_tensors=ignored,
                    projection_parameters=self.metadata['projection_parameters'])


class E2VIDPlusEncoder(_ReleasedBackbone, E2VIDEncoder):
    """Released E2VID+ U-Net with no normalization and explicit ConvLSTM state.

    This is the E2VID architecture family with different normalization/training,
    not a new intrinsically stronger randomly initialized encoder. The '+'
    improvements depend substantially on sim-to-real data and pretrained weights.
    """

    backbone_name = 'E2VID+'

    def __init__(self, bins=5, channels=32):
        nn.Module.__init__(self)
        self._configure_output(bins, channels)
        widths = [32, 64, 128, 256]
        self.head = _ConvLayer(bins, 32, norm=False)
        self.encoders = nn.ModuleList(
            _RecurrentConvLayer(a, b, norm=False) for a, b in zip(widths[:-1], widths[1:]))
        self.resblocks = nn.ModuleList(_ResidualBlock(256, norm=False) for _ in range(2))
        self.decoders = nn.ModuleList(
            _UpsampleConvLayer(c, c // 2, norm=False) for c in reversed(widths[1:]))

    def forward(self, voxel, states=None):
        feature, new_states = super().forward(voxel, states)
        return self.projection(feature), new_states


class _PositionEncoding(nn.Module):
    """The official 1D flattened-token sine encoding, vectorized only.

    The published implementation has 20,000 positions. Keep that explicit limit
    instead of silently repeating/interpolating embeddings at larger resolutions.
    """

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


class ETNetEncoder(_ReleasedBackbone, nn.Module):
    """Full published ET-Net (mls_tpa), minus its final image prediction layer.

    Three ConvLSTMs feed three separate 256-dimensional transformer encoders
    (3 layers each) and decoders (2 layers each, 8 attention heads). They are
    fused exactly as upstream before the three U-Net upsampling stages.

    Attention is global on ceil(H/8)*ceil(W/8) tokens, so memory is quadratic
    in token count. A 128x128 crop has 256 tokens. No hidden recurrent state is
    stored; pass states=None for each new context and carry returned states
    only within that context. States preserve autograd through time.
    """

    backbone_name = 'ET-Net'

    def __init__(self, bins=5, channels=32):
        super().__init__()
        self._configure_output(bins, channels)
        self.head = _ConvLayer(bins, 32, norm=False)
        self.DownsampleConv = nn.ModuleList([
            _RecurrentConvLayer(32, 64, norm=False),
            _RecurrentConvLayer(64, 128, norm=False),
            _RecurrentConvLayer(128, 256, norm=False),
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
            _UpsampleConvLayer(256, 128, norm=False),
            _UpsampleConvLayer(128, 64, norm=False),
            _UpsampleConvLayer(64, 32, norm=False),
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
        return self.projection(feature), tuple(new_states)
