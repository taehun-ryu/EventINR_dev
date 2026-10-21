"""ecoco clips and EVREAL packets: native five-bin inputs with grayscale targets."""

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DATA_RECIPE = 'e2vidplus_original_h5'
NPY_AUGMENTATION_RECIPE = 'e2vid_npy_crop_flip_pause_v2'


def read_manifest(filename):
    filename = Path(filename).resolve()
    lines = filename.read_text(encoding='utf-8-sig').splitlines()
    return [(filename.parent / line.strip()).resolve() for line in lines
            if line.strip() and not line.lstrip().startswith('#')]


class Sequence:
    def __init__(self, path, load_flow=False):
        self.path = Path(path)
        self.h5 = None
        if self.path.suffix.lower() in ('.h5', '.hdf5'):
            from h5_data import H5Sequence
            self.h5 = H5Sequence(self.path, load_flow=load_flow)
            for name in ('times', 'time_origin', 'shape', 'frame_ids'):
                setattr(self, name, getattr(self.h5, name))
            return
        folder = self.path / 'VoxelGrid-betweenframes-5'
        frames = np.loadtxt(self.path / 'frames/timestamps.txt', ndmin=2)
        voxels = np.loadtxt(folder / 'timestamps.txt', ndmin=2)
        ends = np.searchsorted(frames[:, 1], voxels[:, 1])
        if (np.any(ends == 0) or np.any(ends >= len(frames)) or np.any(np.diff(ends) != 1)
                or not np.allclose(frames[ends, 1], voxels[:, 1], rtol=0, atol=1e-8)):
            raise ValueError(f'{path}: voxels must match consecutive GT interval endpoints.')
        selected = frames[ends[0] - 1:ends[-1] + 1]
        self.frame_ids = selected[:, 0].astype(np.int64)
        self.times = selected[:, 1].astype(np.float64)
        self.time_origin = float(self.times[0])
        if np.any(np.diff(self.times) <= 0):
            raise ValueError(f'{path}: GT timestamps must increase strictly.')
        self.images = [self.path / f'frames/frame_{int(i):010d}.png' for i in selected[:, 0]]
        self.voxels = [folder / f'event_tensor_{int(i):010d}.npy' for i in voxels[:, 0]]
        with Image.open(self.images[0]) as image:
            self.shape = (image.height, image.width)
        raw = np.load(self.voxels[0], allow_pickle=False)
        count = raw.sum(0)
        if raw.shape != (5, *self.shape) or not np.allclose(count, count.round(), atol=1e-3, rtol=0):
            raise ValueError(f'{path}: expected original unnormalized five-bin event voxels.')
        if load_flow:
            folder = self.path / 'flow'
            table = np.loadtxt(folder / 'boundary_timestamps.txt', ndmin=2)
            boundaries = {int(row[0]): row[1:3] for row in table}
            actual = np.array([boundaries[index] for index in self.frame_ids[1:]])
            expected = np.stack((self.times[:-1], self.times[1:]), -1)
            tolerance = np.maximum(1e-9, 2 * np.abs(np.spacing(expected)))
            if np.any(np.abs(actual - expected) > tolerance):
                raise ValueError(f'{path}: flow boundaries must match the paired GT timestamps.')
            self.flow_paths = [folder / f'disp01_{index:010d}.npy' for index in self.frame_ids[1:]]

    def __len__(self):
        return len(self.h5) if self.h5 is not None else len(self.voxels)

    def read(self, first, steps):
        if self.h5 is not None:
            return self.h5.read(first, steps)
        voxels = np.stack([np.load(p, allow_pickle=False) for p in self.voxels[first:first + steps]]).astype(np.float32)
        frames = []
        for path in self.images[first:first + steps + 1]:
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert('L'), dtype=np.float32) / 255.)
        times = self.times[first:first + steps + 1]
        return voxels, times, np.stack(frames)

    def read_flow(self, first, steps):
        """Stored disp01 [S-1,2,H,W], previous-to-current pixel displacement."""
        if self.h5 is not None:
            return self.h5.read_flow(first, steps)
        paths = self.flow_paths[first + 1:first + steps]
        flow = (np.stack([np.load(path, allow_pickle=False) for path in paths]).astype(np.float32)
                if paths else np.empty((0, 2, *self.shape), np.float32))
        if flow.shape != (steps - 1, 2, *self.shape) or not np.isfinite(flow).all():
            raise ValueError(f'{self.path}: expected finite pixel displacement [S-1,2,H,W].')
        return flow


def normalize_voxels(voxel, mode='none'):
    """Encoder-only, per-interval nonzero mean/std."""
    if mode == 'none':
        return voxel
    if mode != 'nonzero':
        raise ValueError(f'Unknown voxel normalization: {mode}')
    result = voxel.copy()
    for grid in result:
        mask = grid != 0
        if mask.any():
            values = grid[mask]
            std = max(float(values.std()), 1e-6)
            grid[mask] = (values - values.mean()) / std
    return result


def _sensor_coordinates(height, width, h, w, top, left):
    """Decoder xy remains in the full sensor chart, not the crop chart."""
    y, x = np.indices((h, w))
    return np.stack((2 * (x + left + .5) / width - 1,
                     2 * (y + top + .5) / height - 1), -1).reshape(-1, 2).astype(np.float32)


def _crop_affine(height, width, h, w, top, left):
    """Map full-sensor xy to the cropped encoder feature grid."""
    return dict(sample_scale=np.asarray((width / w, height / h), dtype=np.float32),
                sample_shift=np.asarray(((width - 2 * left - w) / w,
                                         (height - 2 * top - h) / h), dtype=np.float32))


def make_clip(sequence, first, steps, crop=128, augment=True, rng=None,
              voxel_normalization='none', temporal=False):
    rng = np.random.default_rng() if augment and rng is None else rng
    if augment:
        return _make_augmented_npy_clip(sequence, first, steps, crop, rng,
                                       voxel_normalization, temporal)
    voxel, times, frames = sequence.read(first, steps)
    height, width = frames.shape[-2:]
    h, w = (crop, crop) if crop else (height, width)
    if h > height or w > width:
        raise ValueError('Crop exceeds the sensor resolution.')
    top, left = (height - h) // 2, (width - w) // 2
    voxel = voxel[..., top:top + h, left:left + w]
    frames = frames[..., top:top + h, left:left + w]
    flow = sequence.read_flow(first, steps)[..., top:top + h, left:left + w] if temporal else None
    xy = _sensor_coordinates(height, width, h, w, top, left)
    affine = _crop_affine(height, width, h, w, top, left)
    arrays = dict(voxel=normalize_voxels(voxel, voxel_normalization), times=times,
                   time_origin=np.asarray(sequence.time_origin, dtype=np.float64), xy=xy,
                   **affine, target=frames.reshape(steps + 1, -1))
    if temporal:
        arrays['flow'] = flow
    return {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()}


def _npy_pause_indices(steps, rng):

    indices = np.zeros(steps, dtype=np.int64)
    paused = np.zeros(steps, dtype=np.bool_)
    for step in range(1, steps):
        probability = 0.9 if paused[step - 1] else 0.05
        paused[step] = rng.random() < probability
        indices[step] = indices[step - 1] + int(not paused[step])
    return indices, paused


def _make_augmented_npy_clip(sequence, first, steps, crop, rng,
                             voxel_normalization='none', temporal=False):

    if steps < 1:
        raise ValueError('Sequence length must be positive.')
    if first < 0 or first + steps > len(sequence):
        raise IndexError('NPY training clip lies outside its packet range.')
    if crop < 0:
        raise ValueError('Crop must be nonnegative.')
    voxel, times, frames = sequence.read(first, steps)
    packet_indices, paused = _npy_pause_indices(steps, rng)
    source_times = np.concatenate((times[:1], times[packet_indices + 1]))
    voxel = voxel[packet_indices].copy()
    voxel[paused] = 0.
    frames = np.concatenate((frames[:1], frames[packet_indices + 1]), axis=0)

    flow = None
    if temporal:
        real_flow = sequence.read_flow(first, steps)
        flow = np.zeros_like(real_flow)
        for step in range(1, steps):
            if not paused[step]:
                flow[step - 1] = real_flow[packet_indices[step] - 1]

    height, width = frames.shape[-2:]
    h, w = (crop, crop) if crop else (height, width)
    if h > height or w > width:
        raise ValueError('Crop exceeds the sensor resolution.')
    top, left = int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1))
    voxel = voxel[..., top:top + h, left:left + w]
    frames = frames[..., top:top + h, left:left + w]
    if temporal:
        flow = flow[..., top:top + h, left:left + w]
    for axis in (-2, -1):
        if rng.integers(2):
            voxel, frames = np.flip(voxel, axis), np.flip(frames, axis)
            if temporal:
                flow = np.flip(flow, axis).copy()
                flow[:, 1 if axis == -2 else 0] *= -1

    xy = _sensor_coordinates(height, width, h, w, top, left)
    arrays = dict(voxel=normalize_voxels(voxel, voxel_normalization), times=times,
                  time_origin=np.asarray(sequence.time_origin, dtype=np.float64),
                  source_times=source_times, xy=xy,
                  **_crop_affine(height, width, h, w, top, left),
                  target=frames.reshape(steps + 1, -1))
    if temporal:
        arrays['flow'] = flow
    return {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()}


def _original_crop_flip(tensor, seed, crop, augment=True):

    if augment:
        random.seed(seed)
    height, width = tensor.shape[-2:]
    h, w = (crop, crop) if crop else (height, width)
    if h > height or w > width:
        raise ValueError('Crop exceeds the sensor resolution.')
    if not augment:
        top, left = (height - h) // 2, (width - w) // 2
    elif width == w and height == h:
        top, left = 0, 0
    else:
        top = random.randint(0, height - h)
        left = random.randint(0, width - w)
    tensor = tensor[..., top:top + h, left:left + w]
    dims = []
    if augment and random.random() < 0.5:
        dims.append(-1)
    if augment and random.random() < 0.5:
        dims.append(-2)
    if dims:
        tensor = torch.flip(tensor, dims=dims)
    return tensor, top, left


def _original_noise(voxel, noise_std=0.1, noise_fraction=1.0):
    noise = noise_std * torch.randn_like(voxel)
    if noise_fraction < 1.0:
        noise.masked_fill_(torch.rand_like(voxel) >= noise_fraction, 0)
    return voxel + noise


def _original_hot_pixels_(voxels, hot_pixel_std=0.1, max_hot_pixel_fraction=0.001):

    fraction = random.uniform(0, max_hot_pixel_fraction)
    voxel = voxels[0]
    count = int(fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (count,))
    y = torch.randint(0, voxel.shape[-2], (count,))
    values = torch.randn(count, dtype=voxel.dtype, device=voxel.device) * hot_pixel_std
    for item in voxels:
        for index in range(count):
            item[..., :, y[index], x[index]] += values[index]


def make_original_h5_clip(sequence, first, steps, crop=112, augment=True):

    if sequence.h5 is None:
        raise ValueError('Original H5 sequence augmentation requires an H5 source.')
    if first < 0 or steps < 1 or first + steps > len(sequence):
        raise IndexError('H5 training clip lies outside its packet range.')
    seed = random.randint(0, 2 ** 32) if augment else None
    voxels, frames, clocks = [], [], []

    def packet(index, paused=False):
        raw_voxel, timestamp, raw_frame = sequence.h5.training_packet(index)
        voxel, top, left = _original_crop_flip(torch.from_numpy(raw_voxel).float(), seed, crop, augment)
        frame, _, _ = _original_crop_flip(torch.from_numpy(raw_frame).float(), seed, crop, augment)
        if paused:
            voxel.fill_(0.0)
        return voxel, frame, timestamp, top, left

    k = 0
    voxel, frame, clock, top, left = packet(first)
    voxels.append(voxel)
    frames.append(frame)
    clocks.append(clock)
    paused = False
    for _ in range(steps - 1):
        probability_pause = 0.9 if paused else 0.05
        paused = augment and np.random.rand() < probability_pause
        if not paused:
            k += 1
        voxel, frame, clock, _, _ = packet(first + k, paused=paused)
        voxels.append(_original_noise(voxel) if augment else voxel)
        frames.append(frame)
        clocks.append(clock)
    if augment:
        _original_hot_pixels_(voxels)

    # Same full-sensor decoder coordinates / crop affine map as make_clip.
    height, width = sequence.shape
    h, w = frames[0].shape[-2:]
    y, x = np.indices((h, w))
    xy = np.stack((2 * (x + left + .5) / width - 1,
                   2 * (y + top + .5) / height - 1), -1).reshape(-1, 2).astype(np.float32)
    clocks = np.asarray(clocks, dtype=np.float64)
    targets = torch.stack(frames)
    return dict(
        voxel=torch.stack(voxels),
        # Ordinal augmentation axis only; not elapsed sensor time.
        times=torch.linspace(-1., 1., steps + 1, dtype=torch.float64),
        time_origin=torch.tensor(-1., dtype=torch.float64),
        source_times=torch.from_numpy(np.concatenate((clocks[:1], clocks))),
        xy=torch.from_numpy(xy),
        sample_scale=torch.tensor((width / w, height / h), dtype=torch.float32),
        sample_shift=torch.tensor(((width - 2 * left - w) / w,
                                   (height - 2 * top - h) / h), dtype=torch.float32),
        target=torch.cat((targets[:1], targets), dim=0).reshape(steps + 1, -1),
    )


class Clips(Dataset):
    """One augmentation switch; each source retains its own original recipe."""

    def __init__(self, sources, steps=40, crop=112, voxel_normalization='none', temporal=False,
                 augment=True):
        if steps < 1:
            raise ValueError('Sequence length must be positive.')
        self.sequences = [Sequence(path, load_flow=temporal) for path in sources]
        if voxel_normalization != 'none' and any(s.h5 is not None for s in self.sequences):
            raise ValueError('Original E2VID+ H5 training requires voxel_normalization=none.')
        if any(s.h5 is None and len(s) < steps for s in self.sequences):
            raise ValueError('Each original sequence must contain at least steps event intervals.')
        # Existing NPY data: stride 1. E2VID+ H5: stride L, omit N <= L.
        self.strides = [steps if s.h5 is not None else 1 for s in self.sequences]
        counts = [0 if s.h5 is not None and len(s) <= steps else (len(s) - steps) // stride + 1
                  for s, stride in zip(self.sequences, self.strides)]
        self.ends = np.cumsum([0] + counts)
        if not self.sequences or self.ends[-1] == 0:
            raise ValueError('No training clips for the requested steps.')
        self.steps, self.crop = steps, crop
        self.voxel_normalization = voxel_normalization
        self.temporal = temporal
        self.augment = augment
        self.data_recipe = (NPY_AUGMENTATION_RECIPE
                            if augment and any(s.h5 is None for s in self.sequences)
                            else DATA_RECIPE)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError('Training clip index is out of bounds.')
        scene = int(np.searchsorted(self.ends, index, side='right') - 1)
        first = int(index - self.ends[scene]) * self.strides[scene]
        sequence = self.sequences[scene]
        if sequence.h5 is not None:
            return make_original_h5_clip(sequence, first, self.steps, self.crop, augment=self.augment)
        rng = (np.random.default_rng(int(torch.randint(2 ** 32, ()).item()))
               if self.augment else None)
        return make_clip(sequence, first, self.steps, self.crop, augment=self.augment, rng=rng,
                         voxel_normalization=self.voxel_normalization, temporal=self.temporal)


class ECDSequence:
    """EVREAL memmaps: packet j = events[index[j-1]:index[j]], packet 0 empty.

    EVREAL evaluates frames 0..N-2. The unused final frame is NOT a context
    endpoint here. S genuine intervals are paired with S+1 boundary queries.
    Stored clocks/indices are used unchanged; encoder inputs remain raw voxels.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.events = {key: np.load(self.path / f'events_{key}.npy', mmap_mode='r')
                       for key in ('ts', 'xy', 'p')}
        self.images = np.load(self.path / 'images.npy', mmap_mode='r')
        self.times = np.load(self.path / 'images_ts.npy').reshape(-1).astype(np.float64)
        self.time_origin = float(self.times[0])
        self.indices = np.load(self.path / 'image_event_indices.npy').reshape(-1)
        if (self.images.ndim != 4 or self.images.shape[-1] != 1 or self.images.dtype != np.uint8
                or len(self.images) != len(self.times) or len(self.indices) != len(self.times)
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)):
            raise ValueError(f'{path}: expected EVREAL uint8 [N,H,W,1] frames and increasing timestamps.')
        count = len(self.events['ts'])
        if (self.indices.dtype.kind not in 'iu' or np.any(np.diff(self.indices.astype(np.int64)) < 0)
                or np.any(self.indices < 0) or np.any(self.indices > count)
                or self.events['xy'].shape != (count, 2) or len(self.events['p']) != count):
            raise ValueError(f'{path}: invalid stored event indices or event shapes.')
        self.shape = self.images.shape[1:3]
        event_times = self.events['ts']
        self.recording_range = (
            min(float(self.times[0]), float(np.asarray(event_times[0]).item())),
            max(float(self.times[-1]), float(np.asarray(event_times[-1]).item())),
        )
        self.times = self.times[:-1]  # Match len(EVREAL.MemMapDataset) = N-1.

    def __len__(self):
        return len(self.times) - 1

    def packet(self, index):
        """Literal EVREAL temporal splatting, including its equal-time fallback."""
        start, stop = self.indices[max(0, index - 1)], self.indices[index]
        voxel = torch.zeros((5, *self.shape), dtype=torch.float32)
        if start == stop:
            return voxel.numpy()
        xy = np.asarray(self.events['xy'][start:stop])
        stamp = np.asarray(self.events['ts'][start:stop]).reshape(-1)
        polarity = np.asarray(self.events['p'][start:stop]).reshape(-1)
        if not np.isin(polarity, (0, 1)).all():
            raise ValueError('EVREAL memmaps require binary 0/1 polarity, not -1/+1.')
        x, y = torch.from_numpy(xy[:, 0].astype(np.int64)), torch.from_numpy(xy[:, 1].astype(np.int64))
        time = torch.from_numpy((stamp - stamp[0]).astype(np.float32))
        weight = torch.from_numpy(polarity.astype(np.float32) * 2 - 1)
        time = torch.linspace(0, 4, len(time)) if time[-1] < 1e-9 else time / time[-1] * 4
        for bin_index in range(5):
            voxel[bin_index].index_put_((y, x), weight * (1 - (time - bin_index).abs()).clamp(min=0), accumulate=True)
        return voxel.numpy()

    def read(self, first, steps):
        voxel = np.stack([self.packet(index) for index in range(first + 1, first + steps + 1)])
        times, frames = self._read_frames(first, steps + 1)
        return voxel, times, frames

    def read_packets(self, first, count):
        """Read EVREAL packet/frame pairs, including the empty packet 0."""
        voxels = np.stack([self.packet(index) for index in range(first, first + count)])
        times, frames = self._read_frames(first, count)
        return voxels, times, frames

    def _read_frames(self, first, count):
        stop = first + count
        return (self.times[first:stop],
                self.images[first:stop, ..., 0].astype(np.float32) / 255)

    def close(self):
        for array in (*self.events.values(), self.images):
            array._mmap.close()
