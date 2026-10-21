"""Self-contained metrics and timestamp-faithful videos for ECD evaluation.

MSE/SSIM follow EVREAL std (float images, no histogram equalization). LPIPS
uses the standalone lpips package instead of EVREAL's PyIQA wrapper. The
optional skimage CLAHE extension and batched arithmetic match event_inr_release.
"""
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch


class ImageMetrics:
    """Mean per-frame scores, without GT fitting or image-file round trips."""

    def __init__(self, protocol='evreal_std', lpips=False, device='cpu', histogram_equalization='none'):
        if protocol != 'evreal_std':
            raise ValueError('Only the evreal_std metric protocol is supported')
        if histogram_equalization not in ('none', 'clahe'):
            raise ValueError('histogram_equalization must be none or clahe')
        self.protocol, self.device = protocol, torch.device(device)
        self.histogram_equalization, self.perceptual = histogram_equalization, None
        if lpips:
            try:
                import lpips as lpips_lib
            except ImportError as error:
                raise ImportError('Install lpips==0.1.4 and a matching torchvision for LPIPS.') from error
            self.perceptual = lpips_lib.LPIPS(net='alex', version='0.1', verbose=False).to(self.device).eval()
            self.perceptual.requires_grad_(False)

    def _equalize(self, values):
        if self.histogram_equalization == 'none':
            return values
        from skimage.exposure import equalize_adapthist
        return np.stack([equalize_adapthist(frame, clip_limit=.01, nbins=256).astype(np.float32)
                         for frame in values])

    def __call__(self, prediction, target):
        return self.sequence([prediction], [target], batch_size=1)

    @torch.no_grad()
    def sequence(self, predictions, targets, batch_size=16):
        """Bounded batches; SSIM filtering is spatial only, never temporal."""
        from scipy.ndimage import gaussian_filter
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('Metric batch_size must be a positive integer')
        count = len(predictions)
        if count == 0 or count != len(targets):
            raise ValueError('Metrics require matching nonempty prediction and target sequences')
        totals = dict.fromkeys(('l1', 'mse', 'psnr_db', 'ssim'), 0.)
        if self.perceptual is not None:
            totals['lpips'] = 0.
        frame_shape = None
        for start in range(0, count, batch_size):
            pred, gt = [np.asarray(values[start:start + batch_size], dtype=np.float32)
                        for values in (predictions, targets)]
            if pred.ndim != 3 or pred.shape != gt.shape or not all(np.isfinite(x).all() for x in (pred, gt)):
                raise ValueError('Metrics require matching finite grayscale [N,H,W] images')
            if frame_shape is not None and pred.shape[1:] != frame_shape:
                raise ValueError('Metric frames must have a consistent image size')
            frame_shape = pred.shape[1:]
            if min(frame_shape) < 11:
                raise ValueError('Image is too small for the fixed SSIM window')
            if any(x.min() < 0 or x.max() > 1 for x in (pred, gt)):
                raise ValueError('Metric intensities must lie in [0,1]; no implicit intensity normalization')
            if self.perceptual is not None and min(frame_shape) < 64:
                raise ValueError('LPIPS evaluation requires images >=64x64')
            pred, gt = self._equalize(pred), self._equalize(gt)
            mse = np.mean((gt - pred) ** 2, axis=(1, 2), dtype=np.float64)

            def smooth(value):
                return gaussian_filter(value, sigma=(0, 1.5, 1.5), mode='reflect', truncate=3.5)

            # Same 11x11 Gaussian population-covariance SSIM as skimage/EVREAL.
            ux, uy = smooth(gt), smooth(pred)
            vx, vy = smooth(gt * gt) - ux * ux, smooth(pred * pred) - uy * uy
            vxy = smooth(gt * pred) - ux * uy
            similarity = ((2 * ux * uy + .01 ** 2) * (2 * vxy + .03 ** 2)
                          / ((ux ** 2 + uy ** 2 + .01 ** 2) * (vx + vy + .03 ** 2)))
            scores = dict(l1=np.abs(pred.astype(np.float64) - gt).mean(axis=(1, 2)), mse=mse,
                          psnr_db=-10 * np.log10(np.maximum(mse, 1e-12)),
                          ssim=similarity[:, 5:-5, 5:-5].mean(axis=(1, 2), dtype=np.float64))
            if self.perceptual is not None:
                tensors = [torch.as_tensor(np.ascontiguousarray(x), device=self.device)[:, None].repeat(1, 3, 1, 1)
                           for x in (pred, gt)]
                scores['lpips'] = self.perceptual(*(2 * x - 1 for x in tensors)).reshape(len(pred), -1).mean(1).cpu().numpy()
            for key, values in scores.items():
                totals[key] += float(np.sum(values, dtype=np.float64))
        return {key: value / count for key, value in totals.items()}

    def settings(self):
        result = dict(protocol=self.protocol + ('+clahe' if self.histogram_equalization == 'clahe' else ''),
                      inputs='float32 grayscale [0,1]', normalization='none',
                      histogram_equalization=self.histogram_equalization, ssim_gaussian_weights=True,
                      ssim_sigma=1.5, ssim_use_sample_covariance=False,
                      data_range=1., psnr='mean per-frame PSNR; exact-match cap 120 dB',
                      torch=torch.__version__, scikit_image=version('scikit-image'))
        if self.histogram_equalization == 'clahe':
            result['clahe'] = dict(applied_to='prediction and GT independently per frame',
                                   kernel_size='skimage default H//8,W//8', clip_limit=.01, nbins=256)
        if self.perceptual is not None:
            result.update(lpips_backend='lpips', lpips_version=version('lpips'), lpips_network='alex',
                          lpips_model_version='0.1', lpips_input_range='[-1,1]', lpips_grayscale='repeat_to_rgb')
        return result


class EvaluationVideos:
    """Three synchronized previews held on a fixed-FPS, real-time grid.

    Output ticks lie in [first timestamp, last timestamp]. The final encoded
    frame occupies one tick, so playback exceeds the source span by <= 1/fps.
    Preview resampling does not generate new model queries or metric inputs.
    """

    def __init__(self, directory, fps=25, event_clip=3, sample_aligned=False):
        if not np.isfinite([fps, event_clip]).all() or min(fps, event_clip) <= 0:
            raise ValueError('Video fps and event_clip must be finite and positive')
        self.directory, self.fps, self.event_clip = Path(directory), float(fps), float(event_clip)
        self.sample_aligned = bool(sample_aligned)
        self.writers, self.previous, self.first, self.last = [], None, None, None
        self.frames, self.samples, self.closed = 0, 0, False

    def _emit(self, timestamp, inclusive=False):
        tick = (timestamp - self.first) * self.fps
        tolerance = 4 * np.spacing(max(1., abs(timestamp), abs(self.first))) * self.fps
        if abs(tick - round(tick)) <= tolerance:
            tick = float(round(tick))
        limit = int(np.floor(tick) if inclusive else np.ceil(tick) - 1)
        while self.frames <= limit:
            for writer, image in zip(self.writers, self.previous):
                writer.append_data(image)
            self.frames += 1

    def push(self, signed_events, prediction, timestamp, ground_truth):
        event, pred, gt = (np.asarray(value) for value in (signed_events, prediction, ground_truth))
        if self.closed or event.ndim != 2 or event.shape != pred.shape or event.shape != gt.shape:
            raise ValueError('Videos require an open writer and matching event/recon/GT [H,W] images')
        if not np.isfinite(timestamp) or not all(np.isfinite(value).all() for value in (event, pred, gt)):
            raise ValueError('Nonfinite video image or timestamp')
        if any(value.min() < 0 or value.max() > 1 for value in (pred, gt)):
            raise ValueError('Video reconstruction and GT must lie in [0,1]')
        if self.last is not None and (timestamp <= self.last or event.shape != self.previous[0].shape[:2]):
            raise ValueError('Video timestamps must increase and image size must remain fixed')
        if self.first is None:
            import imageio.v2 as imageio
            self.directory.mkdir(parents=True, exist_ok=True)
            self.first = float(timestamp)
            try:
                for name in ('events.mp4', 'recon.mp4', 'gt.mp4'):
                    self.writers.append(imageio.get_writer(self.directory / name, fps=self.fps,
                        codec='libx264', macro_block_size=1,
                        pixelformat='yuv420p' if all(size % 2 == 0 for size in event.shape) else 'yuv444p'))
            except Exception:
                for writer in self.writers:
                    writer.close()
                raise
        elif not self.sample_aligned:
            self._emit(float(timestamp))
        alpha = np.minimum(np.abs(event) / self.event_clip, 1.)[..., None]
        foreground = np.zeros((*event.shape, 3), dtype=np.float32)
        foreground[..., 0], foreground[..., 2] = event > 0, event < 0
        rgb = np.round(255 * (1 - alpha + alpha * foreground)).astype(np.uint8)
        grays = [np.repeat(np.round(value * 255).astype(np.uint8)[..., None], 3, axis=-1) for value in (pred, gt)]
        self.previous, self.last = (rgb, *grays), float(timestamp)
        self.samples += 1
        if self.sample_aligned:
            for writer, image in zip(self.writers, self.previous):
                writer.append_data(image)
            self.frames += 1

    def close(self):
        if not self.closed:
            try:
                if self.previous is not None and not self.sample_aligned:
                    self._emit(self.last, inclusive=True)
            finally:
                self.closed = True
                for writer in self.writers:
                    writer.close()
        resampling = (
            'none; exactly one encoded frame per pushed source sample'
            if self.sample_aligned else
            'previous-sample hold on k/fps grid anchored at first selected GT; ticks within source support'
        )
        return dict(files=['events.mp4', 'recon.mp4', 'gt.mp4'], fps=self.fps, frames=self.frames,
            source_samples=self.samples, source_time_range_s=[self.first, self.last],
            playback_duration_s=self.frames / self.fps, event_clip=self.event_clip,
            sample_aligned=self.sample_aligned, resampling=resampling,
            events='sum of unnormalized voxel bins; positive red, negative blue; opposite polarities can cancel',
            ground_truth='paired reference frames on the same preview grid; no synthesized intermediate GT',
            quantitative_scores=False)

