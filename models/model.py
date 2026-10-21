"""Streaming recurrent features with sigmoid or direct linear intensity readout."""

import math

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .backbone import E2VIDEncoder, E2VIDPlusEncoder, ETNetEncoder
else:
    from backbone import E2VIDEncoder, E2VIDPlusEncoder, ETNetEncoder


DECODER_INPUTS = 35  # Legacy x,y,t input; selected queries use 32 + coord_dim.
DECODER_WIDTHS = (256, 256, 256, 256)
DECODER_OUTPUTS = 1
SIREN_FIRST_OMEGA = 30.
SIREN_HIDDEN_OMEGA = 30.

class SineLayer(nn.Module):
    def __init__(self, inputs, outputs, omega=SIREN_HIDDEN_OMEGA, first=False):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.omega = omega
        bound = 1 / inputs if first else math.sqrt(6 / inputs) / omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, value):
        return (self.omega * self.linear(value)).sin()


def sensor_log(log_intensity, log_eps=0.):
    """Stable log(exp(L) + eps); zero is ideal, a physical floor needs calibration."""
    value = log_intensity
    if value.dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    if log_eps == 0:
        return value
    return torch.logaddexp(value, value.new_tensor(math.log(log_eps)))


class EventINR(nn.Module):

    time_modes = ('global', 'feature', 'local', 'alpha')
    decoder_types = ('siren', 'relu_fourier')

    def __init__(self, encoder='etnet', decoder_widths=DECODER_WIDTHS,
                 time_mode='local', feature_norm='none', decoder_type='siren',
                 intensity_head='sigmoid', coord_dim=None, fourier_bands=7):
        super().__init__()
        if time_mode not in self.time_modes:
            raise ValueError("time_mode must be 'global', 'feature', 'local' or 'alpha'.")
        if decoder_type not in self.decoder_types:
            raise ValueError("decoder_type must be 'siren' or 'relu_fourier'.")
        if intensity_head not in ('sigmoid', 'linear'):
            raise ValueError("intensity_head must be 'sigmoid' or 'linear'.")
        if intensity_head == 'linear' and decoder_type != 'relu_fourier':
            raise ValueError('The linear-intensity comparison uses decoder_type=relu_fourier.')
        coord_dim = (2 if time_mode == 'feature' else 3) if coord_dim is None else coord_dim
        if coord_dim not in (1, 2, 3):
            raise ValueError('coord_dim must be 1 (t), 2 (xy), or 3 (xyt).')
        if time_mode == 'feature' and coord_dim != 2:
            raise ValueError('time_mode=feature omits decoder time and requires coord_dim=2; use local for t queries.')
        if not isinstance(fourier_bands, int) or fourier_bands < 1:
            raise ValueError('fourier_bands must be a positive integer.')
        self.intensity_head = intensity_head
        self.coord_dim = coord_dim
        self.fourier_bands = fourier_bands
        encoders = {'e2vid': E2VIDEncoder, 'e2vidplus': E2VIDPlusEncoder,
                    'etnet': ETNetEncoder}
        self.encoder = encoders[encoder]()
        self.time_mode = time_mode
        self.decoder_type = decoder_type
        norms = {'none': nn.Identity(), 'layer': nn.LayerNorm(32, elementwise_affine=False)}
        self.feature_norm = norms[feature_norm]

        inputs = 32 + coord_dim
        if decoder_type == 'siren':
            # Preserve the legacy parameter layout and random initialization order.
            widths = (inputs, *decoder_widths)
            layers = [SineLayer(source, target,
                                omega=SIREN_FIRST_OMEGA if index == 0 else SIREN_HIDDEN_OMEGA,
                                first=index == 0)
                      for index, (source, target) in enumerate(zip(widths, widths[1:]))]
        else:
            # On [-1, 1], bands span 1 through 2**(fourier_bands-1) cycles.
            frequencies = math.pi * 2. ** torch.arange(fourier_bands, dtype=torch.float32)
            self.register_buffer('coordinate_frequencies', frequencies, persistent=False)
            widths = (32 + 2 * coord_dim * len(frequencies), *decoder_widths)
            layers = []
            for source, target in zip(widths, widths[1:]):
                linear = nn.Linear(source, target)
                nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
                nn.init.zeros_(linear.bias)
                layers.extend((linear, nn.ReLU()))
        output = nn.Linear(widths[-1], DECODER_OUTPUTS)
        bound = math.sqrt(6 / widths[-1]) / SIREN_HIDDEN_OMEGA
        with torch.no_grad():
            output.weight.uniform_(-bound, bound)
            if decoder_type == 'relu_fourier':
                output.bias.zero_()
        self.decoder = nn.Sequential(*layers, output)

    def encode(self, voxels, times, states=None, *, time_origin):
        """Each voxel owns a feature timestamp; raw source clocks may be metadata."""
        if times.dtype != torch.float64 or time_origin.dtype != torch.float64:
            raise ValueError('Use float64 timestamps and sequence time origins.')
        if times.shape != voxels.shape[:2]:
            raise ValueError('Each voxel requires its own right-boundary timestamp.')
        if times.shape[1] == 0 or not torch.isfinite(times).all() or torch.any(times[:, 1:] <= times[:, :-1]):
            raise ValueError('Feature timestamps must be finite and strictly increasing.')
        time_origin = time_origin.reshape(-1, 1)
        if time_origin.shape != (len(voxels), 1) or torch.any(time_origin > times[:, :1]):
            raise ValueError('time_origin must contain one sequence origin per batch item.')

        features = []
        for voxel in voxels.unbind(1):
            feature, states = self.encoder(voxel, states)
            features.append(feature)

        context = dict(features=torch.stack(features, 1), times=times, time_origin=time_origin)
        return context, states

    def query(self, context, xy, times):
        """Return [B,T,Q]: log-sigmoid output or unconstrained linear intensity."""
        if times.dtype != torch.float64:
            raise ValueError('Use float64 query timestamps to preserve clock precision.')
        features, clock = context['features'], context['times']
        if torch.any((times < clock[:, :1]) | (times > clock[:, -1:])):
            raise ValueError('Query support is [t1,tS], not [t0,tS].')

        batch, steps, channels = features.shape[:3]
        xy = xy.to(features.dtype)
        sample_xy = xy
        if 'sample_scale' in context:
            scale = context['sample_scale'].to(features.dtype)[:, None]
            shift = context['sample_shift'].to(features.dtype)[:, None]
            sample_xy = xy * scale + shift
        grid = sample_xy[:, None, None].expand(-1, steps, -1, -1, -1)
        grid = grid.reshape(batch * steps, 1, -1, 2)
        sampled = F.grid_sample(features.flatten(0, 1), grid, mode='bilinear',
                                padding_mode='border', align_corners=False)
        sampled = sampled.reshape(batch, steps, channels, -1).transpose(2, 3)

        right = torch.searchsorted(
            clock.contiguous(), times.contiguous()).clamp(max=steps - 1)
        left = (right - 1).clamp(min=0)
        t_left, t_right = clock.gather(1, left), clock.gather(1, right)
        span = torch.where(right > left, t_right - t_left, torch.ones_like(t_right))
        weight = ((times - t_left) / span).to(features.dtype)[..., None, None]

        shape = (-1, -1, xy.shape[1], channels)
        left_feature = sampled.gather(1, left[..., None, None].expand(shape))
        right_feature = sampled.gather(1, right[..., None, None].expand(shape))
        feature = (1 - weight) * left_feature + weight * right_feature
        feature = self.feature_norm(feature)  # Normalize channels, not coordinates or time.

        spatial = xy[:, None].expand(-1, times.shape[1], -1, -1)
        temporal = None
        temporal_mode = self.time_mode if self.coord_dim != 2 else None
        if temporal_mode == 'global':
            elapsed = (times - context['time_origin']).to(features.dtype)
            temporal = elapsed[..., None, None].expand(-1, -1, xy.shape[1], -1)
        elif temporal_mode == 'local':
            # Fixed clip bounds: querying a subset must not change its time coordinate.
            normalized = (2 * (times - clock[:, :1]) / (clock[:, -1:] - clock[:, :1]) - 1
                          if steps > 1 else torch.zeros_like(times))
            temporal = normalized.to(features.dtype)[..., None, None].expand(-1, -1, xy.shape[1], -1)
        elif temporal_mode == 'alpha':
            # Original local mode: phase of the enclosing feature interval.
            alpha = weight[..., 0, 0]
            alpha = torch.where(right > left, alpha, torch.ones_like(alpha))
            temporal = (2 * alpha - 1)[..., None, None].expand(-1, -1, xy.shape[1], -1)
        coordinates = (temporal if self.coord_dim == 1 else spatial if self.coord_dim == 2
                       else torch.cat((spatial, temporal), -1))
        if self.decoder_type == 'relu_fourier':
            frequencies = self.coordinate_frequencies.to(dtype=coordinates.dtype)
            phase = coordinates.unsqueeze(-1) * frequencies
            coordinates = torch.cat((phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)
        decoder_input = torch.cat((feature, coordinates), -1)
        raw = self.decoder(decoder_input).squeeze(-1)
        # Same parameters/initialization in both arms. No bias/scale adjustment.
        return raw if self.intensity_head == 'linear' else F.logsigmoid(raw)

    def forward(self, voxels, times, xy, query_times, *, time_origin):
        context, _ = self.encode(voxels, times, time_origin=time_origin)
        return self.query(context, xy, query_times)

    @torch.no_grad()
    def render(self, context, times, height=None, width=None):
        """Render decoder outputs [B,T,H,W]; convert using the saved intensity_head."""
        features = context['features']
        height = features.shape[-2] if height is None else height
        width = features.shape[-1] if width is None else width

        y = torch.arange(height, device=features.device, dtype=features.dtype)
        x = torch.arange(width, device=features.device, dtype=features.dtype)
        y = (y + .5) * (2 / height) - 1
        x = (x + .5) * (2 / width) - 1

        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xy = torch.stack((xx, yy), -1).reshape(1, -1, 2)
        xy = xy.expand(features.shape[0], -1, -1)
        return self.query(context, xy, times).reshape(
            features.shape[0], times.shape[1], height, width)
