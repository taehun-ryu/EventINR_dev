"""Linear-intensity Charbonnier / LPIPS and optional relative temporal consistency."""

import torch

from torch.nn import functional as F


class PerceptualLoss(torch.nn.Module):
    """Released E2VID+ style LPIPS; batch/time mean, no prediction clamp."""

    def __init__(self, net='vgg'):
        super().__init__()
        import lpips
        self.perceptual = lpips.LPIPS(net=net, version='0.1', verbose=False)
        self.perceptual.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.perceptual.eval()
        return self

    def forward(self, prediction, target):
        shape = (-1, 1, *prediction.shape[-2:])
        prediction = prediction.reshape(shape).repeat(1, 3, 1, 1)
        target = target.detach().reshape(shape).repeat(1, 3, 1, 1)
        return self.perceptual(prediction, target, normalize=True).mean()


def to_intensity(output, intensity_head='sigmoid'):
    """Linear-head values stay raw, including negative and >1 predictions."""
    if output.dtype in (torch.float16, torch.bfloat16):
        output = output.float()
    if intensity_head == 'linear':
        return output
    if intensity_head == 'sigmoid':
        return output.exp()  # Model returns log(sigmoid(raw)) in the baseline.
    raise ValueError(f'Unknown intensity head: {intensity_head}')


def reconstruction_loss(outputs, target, loss='charbonnier', charbonnier_eps=1e-3,
                        criterion=None, image_shape=None, intensity_head='sigmoid'):
    prediction = to_intensity(outputs, intensity_head)
    if loss == 'lpips':
        shape = (*outputs.shape[:2], 1, *image_shape)
        return criterion(prediction.reshape(shape), target.reshape(shape))

    residual = prediction - target
    charbonnier = (residual.square() + charbonnier_eps ** 2).sqrt()
    return (charbonnier - charbonnier_eps).mean()


def temporal_consistency(prediction, target, flow, start=2, alpha=50.):
    """GT-masked relative warp error; mean over batch, time and image pixels."""
    if prediction.shape[1] <= start:
        return prediction.sum() * 0
    previous = prediction[:, start - 1:-1].flatten(0, 1)
    current = prediction[:, start:].flatten(0, 1)
    height, width = prediction.shape[-2:]

    with torch.no_grad():
        y = torch.arange(height, device=prediction.device, dtype=prediction.dtype)
        x = torch.arange(width, device=prediction.device, dtype=prediction.dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xx, yy), -1)[None]
        grid = grid + flow[:, start - 1:].flatten(0, 1).permute(0, 2, 3, 1).to(prediction.dtype)
        scale = grid.new_tensor((max(width - 1, 1), max(height - 1, 1)))
        grid = 2 * grid / scale - 1
        gt_previous = target[:, start - 1:-1].detach().flatten(0, 1)
        gt_current = target[:, start:].detach().flatten(0, 1)
        warped_gt = F.grid_sample(gt_previous, grid,
                                    padding_mode='zeros', align_corners=True)
        visibility = (-alpha * (gt_current - warped_gt).square()).exp()

    warped = F.grid_sample(previous, grid, padding_mode='zeros', align_corners=True)
    error = (current - warped).abs()
    error = error / (current.abs() + warped.abs() + 1e-5)
    return (visibility * error).mean()


def training_loss(outputs, target, flow=None, loss='charbonnier', charbonnier_eps=1e-3,
                  criterion=None, temporal_weight=0., temporal_start=2, image_shape=None,
                  intensity_head='sigmoid'):
    recon = reconstruction_loss(outputs, target, loss, charbonnier_eps, criterion,
                                image_shape, intensity_head)
    temporal = outputs.new_zeros(())
    if temporal_weight:
        shape = (*outputs.shape[:2], 1, *image_shape)
        temporal = temporal_consistency(to_intensity(outputs, intensity_head).reshape(shape), target.reshape(shape),
                                        -flow, start=temporal_start)
    return recon + temporal_weight * temporal, recon, temporal
