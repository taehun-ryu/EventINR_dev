"""Read ECOCO H5 directly; build native five-bin voxels only for requested clips.

Training packets use the stored event_idx literally, including packet 0.
Evaluation read() keeps half-open GT intervals: event_idx is only a hint,
verified against raw clocks, with binary-search fallback and no full scan.
Raw clocks must already be finite and sorted. Fewer than three events yield
zeros. For 3+ simultaneous events, evaluation uses bin 0; original training
splatting has an undefined denominator, so training rejects that input.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import torch


def event_index(stamps, timestamp, hint=None):
    """Exact left bound: verify a stored index, otherwise binary-search H5."""
    count = len(stamps)
    if isinstance(hint, (int, np.integer)) and not isinstance(hint, (bool, np.bool_)):
        for index in (int(hint), int(hint) + 1):
            if (0 <= index <= count
                    and (index == 0 or float(stamps[index - 1]) < timestamp)
                    and (index == count or timestamp <= float(stamps[index]))):
                return index
    left, right = 0, count
    while left < right:
        middle = (left + right) // 2
        if float(stamps[middle]) < timestamp:
            left = middle + 1
        else:
            right = middle
    return left


def voxel_grid(events, begin, end, shape, chunk=1_000_000):
    voxel = np.zeros((5, *shape), dtype=np.float32)
    if end - begin < 3:
        return voxel
    first = float(np.asarray(events['ts'][begin]).item())
    last = float(np.asarray(events['ts'][end - 1]).item())
    duration = np.float32(last - first)
    for start in range(begin, end, chunk):
        stop = min(start + chunk, end)
        x = np.asarray(events['xs'][start:stop]).reshape(-1)
        y = np.asarray(events['ys'][start:stop]).reshape(-1)
        p = np.asarray(events['ps'][start:stop]).reshape(-1)
        if (not np.isin(p, (0, 1)).all() or np.any(x != np.floor(x)) or np.any(y != np.floor(y))
                or np.any((x < 0) | (x >= shape[1]) | (y < 0) | (y >= shape[0]))):
            raise ValueError('H5 events require integer sensor coordinates and binary 0/1 polarity.')
        x, y = x.astype(np.int64), y.astype(np.int64)
        relative = (np.asarray(events['ts'][start:stop], dtype=np.float64).reshape(-1) - first).astype(np.float32)
        time = relative / duration * 4 if duration > 0 else np.zeros_like(relative)
        polarity = p.astype(np.float32) * 2 - 1
        for bin_index in range(5):
            weight = np.maximum(0, 1 - np.abs(time - bin_index)) * polarity
            np.add.at(voxel[bin_index], (y, x), weight)
    return voxel


def training_voxel_grid(events, begin, end, shape):
    """E2VID+ events_to_voxel_torch: five unnormalized temporal bins on CPU."""
    voxel = torch.zeros((5, *shape), dtype=torch.float32)
    if end - begin < 3:
        return voxel.numpy()
    x = np.asarray(events['xs'][begin:end]).reshape(-1)
    y = np.asarray(events['ys'][begin:end]).reshape(-1)
    p = np.asarray(events['ps'][begin:end]).reshape(-1)
    ts = np.asarray(events['ts'][begin:end], dtype=np.float64).reshape(-1)
    if (not np.isin(p, (0, 1)).all() or np.any(x != np.floor(x)) or np.any(y != np.floor(y))
            or np.any((x < 0) | (x >= shape[1]) | (y < 0) | (y >= shape[0]))):
        raise ValueError('H5 events require integer sensor coordinates and binary 0/1 polarity.')
    time = torch.from_numpy((ts - ts[0]).astype(np.float32))
    if not torch.isfinite(time).all() or torch.any(time[1:] < time[:-1]) or time[-1] <= 0:
        raise ValueError('E2VID+ temporal splatting needs finite sorted events with positive duration for 3+ events.')
    time = (time - time[0]) / (time[-1] - time[0]) * 4
    x = torch.from_numpy(x.astype(np.float32)).long()
    y = torch.from_numpy(y.astype(np.float32)).long()
    polarity = torch.from_numpy((p * 2.0 - 1.0).astype(np.float32))
    zeros = torch.zeros_like(time)
    for bin_index in range(5):
        weight = polarity * torch.maximum(zeros, 1.0 - torch.abs(time - bin_index))
        voxel[bin_index].index_put_((y, x), weight, accumulate=True)
    return voxel.numpy()


class H5Sequence:
    def __init__(self, path, load_flow=False):
        self.path = Path(path)
        self._file, self._pid = None, None
        if load_flow:
            raise ValueError('H5 TC requires verified interval displacement flow; raw ESIM flow is not assumed compatible.')
        with h5py.File(self.path, 'r') as file:
            self.keys = sorted(file['images'])
            self.times, self.hints = [], []
            for key in self.keys:
                attributes = file['images'][key].attrs
                self.times.append(attributes['timestamp'])
                self.hints.append(attributes.get('event_idx'))
            self.times = np.asarray(self.times, dtype=np.float64)
            if len(self.times) < 2 or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
                raise ValueError(f'{path}: expected at least two strictly increasing GT timestamps.')
            image = file['images'][self.keys[0]]
            self.shape = image.shape[:2]
            valid_shape = len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 1)
            if not valid_shape or min(self.shape) < 1:
                raise ValueError(f'{path}: expected grayscale GT [H,W] or [H,W,1].')
            self.image_shape = image.shape
            if image.dtype != np.uint8:
                raise ValueError(f'{path}: GT must have consistent uint8 grayscale shape.')
            count = len(file['events/ts'])
            if any(file[f'events/{key}'].size != count for key in ('ts', 'xs', 'ys', 'ps')):
                raise ValueError(f'{path}: event array lengths disagree.')
            self.indices = np.full(len(self.times), -1, dtype=np.int64)
        self.frame_ids = np.arange(len(self.keys), dtype=np.int64)
        self.time_origin = float(self.times[0])

    def __len__(self):
        return len(self.times) - 1

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'], state['_pid'] = None, None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file, self._pid = None, None

    def _open(self):
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.path, 'r')
            self._pid = os.getpid()
        return self._file

    def training_packet(self, index):
        """Literal E2VID+ H5 item: stored indices, packet 0 starts at event 0."""
        if not 0 <= index < len(self):
            raise IndexError('H5 training packet lies outside the E2VID+ frame range.')
        file = self._open()
        events = {key: file[f'events/{key}'] for key in ('ts', 'xs', 'ys', 'ps')}
        begin = 0 if index == 0 else self.hints[index - 1]
        end = self.hints[index]
        if (not isinstance(begin, (int, np.integer)) or not isinstance(end, (int, np.integer))
                or isinstance(begin, (bool, np.bool_)) or isinstance(end, (bool, np.bool_))
                or not 0 <= begin <= end <= len(events['ts'])):
            raise ValueError(f'{self.path}: original E2VID+ loading requires valid stored event_idx values.')
        voxel = training_voxel_grid(events, int(begin), int(end), self.shape)
        image = file['images'][self.keys[index]]
        if image.shape != self.image_shape or image.dtype != np.uint8:
            raise ValueError(f'{self.path}: GT must have consistent uint8 grayscale shape.')
        frame = np.asarray(image).reshape(self.shape).astype(np.float32) / 255
        return voxel, self.times[index], frame

    def read(self, first, steps):
        if first < 0 or steps < 1 or first + steps > len(self):
            raise IndexError('H5 clip lies outside its GT interval range.')
        file = self._open()
        events = {key: file[f'events/{key}'] for key in ('ts', 'xs', 'ys', 'ps')}
        for index in range(first, first + steps + 1):
            if self.indices[index] < 0:
                self.indices[index] = event_index(events['ts'], self.times[index], self.hints[index])
        voxels = np.stack([voxel_grid(events, int(self.indices[index]), int(self.indices[index + 1]), self.shape)
                           for index in range(first, first + steps)])
        frames = []
        for key in self.keys[first:first + steps + 1]:
            image = file['images'][key]
            if image.shape != self.image_shape or image.dtype != np.uint8:
                raise ValueError(f'{self.path}: GT must have consistent uint8 grayscale shape.')
            frames.append(np.asarray(image).reshape(self.shape).astype(np.float32) / 255)
        return voxels, self.times[first:first + steps + 1], np.stack(frames)

    def read_flow(self, first, steps):
        raise ValueError('H5 TC requires verified interval displacement flow, not uncertified raw ESIM flow.')
