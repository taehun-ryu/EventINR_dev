"""Streaming recurrent features and a coordinate SIREN for normalized log intensity."""

import math

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .backbone import E2VIDEncoder, E2VIDPlusEncoder, ETNetEncoder
else:
    from backbone import E2VIDEncoder, E2VIDPlusEncoder, ETNetEncoder


DECODER_INPUTS = 35  # 32 feature channels plus (x, y, time); feature-only uses 34.
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
    """Voxel Vi produces Fi at its right boundary ti, for i=1,...,S.

    Spatial bilinear sampling and actual-clock linear interpolation provide a
    feature F(x,y,t). ``local`` also gives the SIREN the query time normalized
    over the entire encoded clip, not the phase of one feature interval.
    ``alpha`` restores that enclosing-interval phase, with +1 at every feature
    endpoint (including the first). ``feature`` omits decoder time entirely.
    ``global`` retains the legacy elapsed clock for checkpoint compatibility.
    """

    time_modes = ('global', 'feature', 'local', 'alpha')

    def __init__(self, encoder='etnet', decoder_widths=DECODER_WIDTHS,
                 time_mode='local', feature_norm='none'):
        super().__init__()
        if time_mode not in self.time_modes:
            raise ValueError("time_mode must be 'global', 'feature', 'local' or 'alpha'.")
        encoders = {'e2vid': E2VIDEncoder, 'e2vidplus': E2VIDPlusEncoder,
                    'etnet': ETNetEncoder}
        self.encoder = encoders[encoder]()
        self.time_mode = time_mode
        norms = {'none': nn.Identity(), 'layer': nn.LayerNorm(32, elementwise_affine=False)}
        self.feature_norm = norms[feature_norm]

        inputs = DECODER_INPUTS - 1 if time_mode == 'feature' else DECODER_INPUTS
        widths = (inputs, *decoder_widths)
        layers = [SineLayer(source, target,
                            omega=SIREN_FIRST_OMEGA if index == 0 else SIREN_HIDDEN_OMEGA,
                            first=index == 0)
                  for index, (source, target) in enumerate(zip(widths, widths[1:]))]
        output = nn.Linear(widths[-1], DECODER_OUTPUTS)
        bound = math.sqrt(6 / widths[-1]) / SIREN_HIDDEN_OMEGA
        with torch.no_grad():
            output.weight.uniform_(-bound, bound)
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
        """xy [B,Q,2], times [B,T] -> log intensity <= 0, shaped [B,T,Q]."""
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
        decoder_input = torch.cat((feature, spatial), -1)
        if self.time_mode == 'global':
            elapsed = (times - context['time_origin']).to(features.dtype)
            temporal = elapsed[..., None, None].expand(-1, -1, xy.shape[1], -1)
            decoder_input = torch.cat((decoder_input, temporal), -1)
        elif self.time_mode == 'local':
            # Fixed clip bounds: querying a subset must not change its time coordinate.
            normalized = (2 * (times - clock[:, :1]) / (clock[:, -1:] - clock[:, :1]) - 1
                          if steps > 1 else torch.zeros_like(times))
            temporal = normalized.to(features.dtype)[..., None, None].expand(-1, -1, xy.shape[1], -1)
            decoder_input = torch.cat((decoder_input, temporal), -1)
        elif self.time_mode == 'alpha':
            # Original local mode: phase of the enclosing feature interval.
            alpha = weight[..., 0, 0]
            alpha = torch.where(right > left, alpha, torch.ones_like(alpha))
            phase = (2 * alpha - 1)[..., None, None].expand(-1, -1, xy.shape[1], -1)
            decoder_input = torch.cat((decoder_input, phase), -1)
        raw = self.decoder(decoder_input).squeeze(-1)
        return F.logsigmoid(raw)  # exp(log I) = sigmoid(raw) is always in (0, 1).

    def forward(self, voxels, times, xy, query_times, *, time_origin):
        context, _ = self.encode(voxels, times, time_origin=time_origin)
        return self.query(context, xy, query_times)

    @torch.no_grad()
    def render(self, context, times, height=None, width=None):
        """Render normalized log frames [B,T,H,W]; call model.eval() for evaluation."""
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
