"""Train EventINR with selectable t/xy/xyt queries and sigmoid or linear intensity."""

import json
import math
import random
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Clips, Sequence, normalize_voxels, read_manifest, DATA_RECIPE
from models.model import EventINR, DECODER_WIDTHS
from losses import PerceptualLoss, training_loss, to_intensity


def save_preview(target, prediction, path):
    """Two display-only views. GT stays [0,1]; metrics use raw prediction."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or prediction.ndim != 3 or not len(prediction):
        raise ValueError('Preview expects matching nonempty [T,H,W] arrays.')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError('Non-finite preview values.')
    path = Path(path)
    indices = sorted(set((0, (len(prediction) - 1) // 2, len(prediction) - 1)))

    def write(values, destination):
        panel = np.concatenate([np.concatenate((target[i], values[i]), 0) for i in indices], 1)
        Image.fromarray((panel.clip(0, 1) * 255).round().astype(np.uint8)).save(destination)

    write(prediction, path)  # Fixed [0,1] clamp for absolute-brightness comparison.
    low, high = map(float, np.percentile(prediction, (1, 99)))
    # One range over the entire preview clip, never a different range per frame.
    stretched = ((prediction - low) / (high - low) if high - low > 1e-8
                 else np.full_like(prediction, .5))
    percentile_path = path.with_name(path.stem + '_percentile' + path.suffix)
    write(stretched, percentile_path)
    return dict(percentile_file=percentile_path.name, percentile_low=low,
                percentile_high=high, percentile_scope='prediction_clip',
                preview_mapping='clamp_0_1', percentile_mapping='p1_p99',
                gt_mapping='unchanged_0_1')


@contextmanager
def isolated_rng(seed=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def make_preview_batch(dataset, seed):
    with isolated_rng(seed):
        sample = dataset[0]
    return {name: value.detach().clone()[None] for name, value in sample.items()}


@torch.no_grad()
def save_training_preview(model, batch, path, charbonnier_eps=1e-3):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with isolated_rng():
            device = next(model.parameters()).device
            sample = {name: value.to(device) for name, value in batch.items()}
            times = sample['times'][:, 1:]
            context, _ = model.encode(sample['voxel'], times, time_origin=sample['time_origin'])
            for name in ('sample_scale', 'sample_shift'):
                if name in sample:
                    context[name] = sample[name]
            outputs = model.query(context, sample['xy'], times)
            prediction = to_intensity(outputs, model.intensity_head)[0].cpu()
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('Non-finite prediction in training preview.')
            target = sample['target'][0, 1:].cpu()
            features = context['features'].detach().float()
            residual, constant = prediction - target, target - target.median()
            stats = dict(
                time_mode=model.time_mode, intensity_head=model.intensity_head,
                coord_dim=model.coord_dim, fourier_bands=model.fourier_bands,
                l1=float(residual.abs().mean()), constant_l1=float(constant.abs().mean()),
                charbonnier=float(((residual.square() + charbonnier_eps ** 2).sqrt() - charbonnier_eps).mean()),
                constant_charbonnier=float(((constant.square() + charbonnier_eps ** 2).sqrt() - charbonnier_eps).mean()),
                prediction_min=float(prediction.min()), prediction_max=float(prediction.max()),
                prediction_mean=float(prediction.mean()),
                out_of_range_fraction=float(((prediction < 0) | (prediction > 1)).float().mean()),
                clamped_l1=float((prediction.clamp(0, 1) - target).abs().mean()),
                prediction_std=float(prediction.std(unbiased=False)), target_std=float(target.std(unbiased=False)),
                feature_std=float(features.std(unbiased=False)), feature_abs_max=float(features.abs().max()),
                file=Path(path).name,
            )
    finally:
        for module, mode in modes.items():
            module.training = mode
    steps, _, height, width = batch['voxel'].shape[1:]
    stats.update(save_preview(target.reshape(steps, height, width).numpy(),
                              prediction.reshape(steps, height, width).numpy(), path))
    return stats


def stream_chunks(first, stop, steps):
    for begin in range(first, stop, steps):
        yield begin, min(steps, stop - begin)


@torch.no_grad()
def stream_sequence(model, sequence, steps, device, voxel_normalization='none', first=0, stop=None):
    """Carry state within a sequence; encode each native voxel once."""
    states = None
    stop = len(sequence) if stop is None else stop
    time_origin = torch.as_tensor([float(sequence.times[0])], device=device, dtype=torch.float64)

    for begin, count in stream_chunks(first, stop, steps):
        voxel, clock, target = sequence.read(begin, count)
        values = torch.as_tensor(normalize_voxels(voxel, voxel_normalization), device=device)[None]
        times = torch.as_tensor(clock[1:], device=device)[None]
        context, states = model.encode(values, times, states, time_origin=time_origin)
        yield begin, voxel, clock, target, model.render(context, times)[0]


def run_epoch(model, loader, optimizer, device, loss='charbonnier',
              charbonnier_eps=1e-3, criterion=None, temporal_weight=0., temporal_start=2):
    model.train()
    totals, examples = np.zeros(3), 0
    output_min, output_max = float('inf'), -float('inf')
    grad_total, grad_max, clipped = 0., 0., 0
    bar = tqdm(loader, total=len(loader), desc='Train', mininterval=10., miniters=1, leave=False)

    for batch in bar:
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        times = batch['times'][:, 1:]
        context, _ = model.encode(batch['voxel'], times, None, time_origin=batch['time_origin'])
        if isinstance(context, dict) and 'sample_scale' in batch:
            context['sample_scale'] = batch['sample_scale']
            context['sample_shift'] = batch['sample_shift']
        outputs = model.query(context, batch['xy'], times)
        objective, recon, temporal = training_loss(
            outputs, batch['target'][:, 1:], batch.get('flow'), loss, charbonnier_eps,
            criterion, temporal_weight, temporal_start, batch['voxel'].shape[-2:],
            intensity_head=model.intensity_head)
        if not torch.isfinite(objective):
            raise FloatingPointError('Non-finite training loss.')

        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()

        size = len(outputs)
        values = torch.stack((objective, recon, temporal, outputs.min(), outputs.max(), grad_norm)).detach().cpu().numpy()
        totals += size * values[:3]
        examples += size
        output_min, output_max = min(output_min, float(values[3])), max(output_max, float(values[4]))
        grad_total += size * float(values[5])
        grad_max = max(grad_max, float(values[5]))
        clipped += size * int(values[5] > 1.)
        bar.set_postfix({name: f'{totals[i] / examples:.5f}'
                         for i, name in enumerate(('loss', 'recon', 'tc'))}, refresh=False)

    return dict(zip(('objective', 'recon', 'temporal_loss'), (totals / examples).tolist()),
                output_min=output_min, output_max=output_max,
                preclip_grad_mean=grad_total / examples, preclip_grad_max=grad_max,
                clip_fraction=clipped / examples)


@torch.no_grad()
def evaluate_sequence(model, sequence, steps, device, preview=None, voxel_normalization='none'):
    model.eval()
    scores, biases = [], []
    for first, _, _, target, outputs in stream_sequence(model, sequence, steps, device, voxel_normalization):
        if not torch.isfinite(outputs).all():
            raise FloatingPointError('Non-finite decoder output.')
        prediction = to_intensity(outputs, model.intensity_head).cpu().numpy()
        for local, image in enumerate(prediction, start=1):
            gt = target[local]
            mse = float(np.square(image.astype(np.float64) - gt).mean())
            l1 = float(np.abs(image - gt).mean())
            psnr = -10 * math.log10(max(mse, 1e-12))
            ssim = structural_similarity(gt, image, data_range=1., gaussian_weights=True,
                                         sigma=1.5, use_sample_covariance=False)
            scores.append((l1, mse, psnr, ssim))
            biases.append(float((image.astype(np.float64) - gt).mean()))

        if first == 0 and preview is not None:
            save_preview(target[1:], prediction, preview)
    return dict(zip(('l1', 'mse', 'psnr_db', 'ssim'), np.mean(scores, axis=0).tolist()),
                brightness_bias=float(np.mean(biases)), frames=len(scores))


def validate(model, sequences, args, device, preview=None):
    rows = [evaluate_sequence(model, sequence, args.steps, device,
                             preview if i == 0 else None, args.voxel_normalization)
            for i, sequence in enumerate(tqdm(sequences, desc='Val', mininterval=10., leave=False))]
    count = sum(row['frames'] for row in rows)
    scores = {key: sum(row[key] * row['frames'] for row in rows) / count
              for key in ('l1', 'mse', 'psnr_db', 'ssim', 'brightness_bias')}
    return dict(scores, frames=count)


def save_checkpoint(path, state):
    temporary = Path(path).with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    # Load only trusted checkpoints: torch.load reads executable pickle data.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    args = saved['args']
    mode = args['time_mode']
    supported = getattr(EventINR, 'time_modes', ('global', 'feature', 'local'))
    if mode == 'local' and 'alpha' not in supported and saved.get('time_coordinate') not in (None, 'alpha'):
        raise ValueError('This checkpoint needs the clip-normalized model.py, not the original interval-local model.')
    if mode == 'local' and saved.get('time_coordinate', 'alpha') == 'alpha' and 'alpha' in supported:
        mode = 'alpha'  # Original time_loss_grid local used interval phase.
    elif mode == 'alpha' and 'alpha' not in supported:
        mode = 'local'
    decoder_type = args.get('decoder_type', 'siren')
    coord_dim = args.get('coord_dim', 2 if mode == 'feature' else 3)
    if coord_dim not in (1, 2, 3) or (mode == 'feature' and coord_dim != 2):
        raise ValueError('Checkpoint has incompatible coord_dim/time_mode metadata.')
    first_key = 'decoder.0.weight' if decoder_type == 'relu_fourier' else 'decoder.0.linear.weight'
    inputs = saved['model'][first_key].shape[1]
    if decoder_type == 'relu_fourier':
        extra, divisor = inputs - 32, 2 * coord_dim
        if extra <= 0 or extra % divisor:
            raise ValueError('Checkpoint decoder width does not match its coordinate dimensions.')
        inferred_bands = extra // divisor
        fourier_bands = args.get('fourier_bands', inferred_bands)
        if not isinstance(fourier_bands, int) or fourier_bands != inferred_bands:
            raise ValueError('Checkpoint fourier_bands metadata does not match decoder weights.')
    else:
        fourier_bands = args.get('fourier_bands', 7)  # Unused by SIREN.
        if inputs != 32 + coord_dim:
            raise ValueError('Checkpoint SIREN width does not match its coordinate dimensions.')
    # Also expose resolved legacy settings in evaluation reports and resume checks.
    saved['args'] = {**args, 'coord_dim': coord_dim, 'fourier_bands': fourier_bands}
    model = EventINR(encoder=args['encoder'], decoder_widths=args['decoder_widths'],
                     time_mode=mode, feature_norm=args.get('feature_norm', 'none'),
                     decoder_type=decoder_type, coord_dim=coord_dim, fourier_bands=fourier_bands,
                     intensity_head=args.get('intensity_head', 'sigmoid')).to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), saved


def checkpoint_state(args, model, optimizer, generator, record):
    return dict(
        args={**vars(args), 'feature_norm': 'none', 'decoder_widths': list(DECODER_WIDTHS),
              'coord_dim': model.coord_dim, 'fourier_bands': model.fourier_bands},
        epoch=record['epoch'], best_value=record['best_value'],
        data_recipe=DATA_RECIPE,
        time_coordinate=('alpha' if model.time_mode == 'local'
                         and 'alpha' not in getattr(model, 'time_modes', ()) else model.time_mode),
        model=model.state_dict(), optimizer=optimizer.state_dict(), metrics=record,
        rng=torch.get_rng_state(), loader_rng=generator.get_state(),
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.device(args.device).type == 'cuda' else [])


def resume_training(args, model, optimizer, generator, dataset, device, *, data_recipe, clip_strides):
    """Check compatibility before restoring weights, optimizer and RNG state."""
    saved_model, saved = load_checkpoint(args.resume, device)
    for name in ('intensity_head', 'coord_dim', 'fourier_bands'):
        if getattr(saved_model, name) != getattr(model, name):
            raise ValueError(f'Resume must preserve {name}; start a fresh run for the comparison.')
    previous = saved['args']
    if previous.get('no_h5_augmentation', False):
        raise ValueError('This checkpoint used mixed NPY/H5 augmentation settings; start a new run for the shared augmentation switch.')
    if saved.get('data_recipe') != data_recipe:
        raise ValueError('Data loading changed; start a new run instead of resuming a different recipe.')
    if previous.get('max_batches', 0):
        raise ValueError('Resume requires full-loader training; start a new run instead of resuming batch-limited training.')
    if previous.get('feature_norm', 'none') != 'none':
        raise ValueError('Resume requires raw encoder features; start a new run instead of resuming feature-normalized training.')
    if saved.get('clip_strides', [args.steps]) != clip_strides or (
            'clip_strides' not in saved and any(s.h5 is None for s in dataset.sequences)):
        raise ValueError('Clip sampling changed; start a new run instead of resuming a different stride.')
    supported = dict(recon_domain='linear', event_weight=0., spatial_weight=0., temporal_mode='relative', points=0)
    if any(previous.get(key, value) != value for key, value in supported.items()):
        raise ValueError('Resume requires the same linear reconstruction / relative TC objective.')
    ignored = {'resume', 'out', 'epochs', 'workers', 'device', 'save_every', 'preview_every'}
    saved_args = {'feature_norm': 'none', 'decoder_type': 'siren',
                  'no_augmentation': False, 'intensity_head': 'sigmoid', **previous}
    if saved_model.time_mode != model.time_mode or any(
            saved_args[key] != value for key, value in vars(args).items() if key not in ignored):
        raise ValueError('Resume must preserve the model, data, and loss settings.')
    model.load_state_dict(saved_model.state_dict())
    del saved_model
    optimizer.load_state_dict(saved['optimizer'])
    epoch, best = saved['epoch'], saved['best_value']
    torch.set_rng_state(saved['rng'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    generator.set_state(saved['loader_rng'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return epoch, best


def main(args):
    if args.coord_dim is None:
        args.coord_dim = 2 if args.time_mode == 'feature' else 3
    if args.time_mode == 'feature' and args.coord_dim != 2:
        raise ValueError('--time_mode feature requires --coord_dim 2; use local for t queries.')
    if args.fourier_bands < 1:
        raise ValueError('--fourier_bands must be positive.')
    if args.voxel_normalization == 'auto':
        args.voxel_normalization = 'nonzero' if args.encoder == 'e2vid' else 'none'
    destination = Path(args.out)
    if not args.resume and destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Choose an empty output directory or use --resume.')
    if args.steps < 1 or (args.temporal_weight and not 1 <= args.temporal_start < args.steps):
        raise ValueError('Use positive steps; active TC requires 1 <= temporal_start < steps.')

    if args.save_every < 0 or args.preview_every < 0:
        raise ValueError('--save_every and --preview_every must be nonnegative.')

    sources = read_manifest(args.train)
    val_sources = read_manifest(args.val) if args.val is not None else None
    if val_sources is not None and (not val_sources or {p.name for p in sources} & {p.name for p in val_sources}):
        raise ValueError('Use nonempty, sequence-disjoint training and validation splits.')
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    val_count = len(val_sources) if val_sources is not None else 0
    print(f'Loading {len(sources)} training / {val_count} validation sequences ...', flush=True)
    dataset = Clips(sources, args.steps, args.crop, args.voxel_normalization,
                    temporal=args.temporal_weight != 0, augment=not args.no_augmentation)
    data_recipe = getattr(dataset, 'data_recipe', DATA_RECIPE)
    clip_strides = sorted(set(dataset.strides))
    validation = [Sequence(path) for path in val_sources] if val_sources is not None else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        generator=generator, pin_memory=device.type == 'cuda')

    # Decoder construction must not change training/dropout/augmentation RNG.
    with isolated_rng():
        model = EventINR(encoder=args.encoder, time_mode=args.time_mode,
                         decoder_type=args.decoder_type, intensity_head=args.intensity_head,
                         coord_dim=args.coord_dim, fourier_bands=args.fourier_bands).to(device)
    criterion = None
    if args.loss == 'lpips':
        with torch.random.fork_rng(devices=[]):
            criterion = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    epoch, best = 0, -float('inf') if validation is not None else None

    if args.resume:
        epoch, best = resume_training(args, model, optimizer, generator, dataset, device,
                                      data_recipe=data_recipe, clip_strides=clip_strides)

    destination.mkdir(parents=True, exist_ok=True)
    preview_batch = (make_preview_batch(dataset, args.seed)
                     if validation is None and args.preview_every else None)
    stride_text = ','.join(map(str, clip_strides))
    print(f'{len(dataset):,} clips; stride={stride_text}; {len(loader):,} batches/epoch; '
          f'{args.encoder} + {args.decoder_type}; intensity_head={model.intensity_head}; '
          f'coords={"t" if model.coord_dim == 1 else "xy" if model.coord_dim == 2 else "xyt"}; '
          f'fourier_bands={model.fourier_bands}; '
          f'time={args.time_mode}; feature_norm=none; '
          f'augmentation={"off" if args.no_augmentation else "on"} (NPY and H5); '
          f'{args.steps} endpoint frames; '
          f'linear {args.loss} + {args.temporal_weight:g} relative TC', flush=True)
    if any(sequence.h5 is not None for sequence in dataset.sequences):
        augmentation_text = ('fixed center crop; no flip/pause/noise/hot-pixel augmentation'
                             if args.no_augmentation else
                             'shared random crop/flip; original pause/noise/hot pixels')
        print(f'E2VID+ H5 loading: stored event_idx; online 5-bin voxels; {augmentation_text}; '
              'source clocks preserved; '
              'INR query axis follows normalized clip step order.', flush=True)
    if any(sequence.h5 is None for sequence in dataset.sequences):
        augmentation_text = ('shared random crop, H/V flips; no rotation; '
                             'virtual pauses .05/.9 from E2VID+ protocol'
                             if not args.no_augmentation else
                             'fixed center crop; no flip/pause augmentation')
        clock_text = ('original clocks and consecutive packets unchanged'
                      if args.no_augmentation else
                      'source clocks retained in source_times; '
                      'INR query axis advances on the original clip time slots')
        print(f'E2VID NPY loading: {augmentation_text}; {clock_text}.', flush=True)
    
    for epoch in tqdm(range(epoch + 1, args.epochs + 1), desc='Epochs', mininterval=10.):
        training = run_epoch(model, loader, optimizer, device, args.loss,
                             args.charbonnier_eps, criterion, args.temporal_weight, args.temporal_start)
        preview = (save_training_preview(model, preview_batch, destination / 'preview.png', args.charbonnier_eps)
                   if preview_batch is not None and epoch % args.preview_every == 0 else None)
        
        scores, improved = None, False
        if validation is not None:
            scores = validate(model, validation, args, device, destination / 'preview.png')
            selected = scores['psnr_db' if args.best_metric == 'psnr' else 'ssim']
            if not math.isfinite(selected):
                raise FloatingPointError('Non-finite validation metric.')
            improved, best = selected > best, max(best, selected)
        record = dict(epoch=epoch, train=training, val=scores,
                      best_metric=args.best_metric if scores is not None else None, best_value=best,
                      training_preview=preview)
        
        state = checkpoint_state(args, model, optimizer, generator, record)
        state['data_recipe'] = data_recipe
        state['clip_strides'] = clip_strides
        save_checkpoint(destination / 'last.pt', state)
        if validation is None and args.save_every and epoch % args.save_every == 0:
            save_checkpoint(destination / f'epoch_{epoch:03d}.pt', state)
        if improved:
            save_checkpoint(destination / 'best.pt', state)

        with (destination / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

        message = f"Epoch {epoch}: recon={training['recon']:.5f}, tc={training['temporal_loss']:.5f}"
        if scores is not None:
            message += (f"; val PSNR={scores['psnr_db']:.3f}, SSIM={scores['ssim']:.4f}, "
                        f"bias={scores['brightness_bias']:+.4f}")
            
        tqdm.write(message)


def parser():
    p = ArgumentParser(description=__doc__)
    for name in ('train', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--val', default=None)
    p.add_argument('--resume')
    p.add_argument('--save_every', type=int, default=5,
                   help='Without validation, retain a checkpoint every N epochs; 0 disables.')
    p.add_argument('--preview_every', type=int, default=1,
                   help='Without validation, update preview.png every N epochs; 0 disables.')
    p.add_argument('--encoder', choices=('e2vid', 'e2vidplus', 'etnet'), default='etnet')
    p.add_argument('--decoder_type', choices=('siren', 'relu_fourier'), default='siren')
    p.add_argument('--coord_dim', type=int, choices=(1, 2, 3), default=None,
                   help='1: t only; 2: xy only; 3: xyt. Default: 3, or 2 in legacy feature mode.')
    p.add_argument('--fourier_bands', type=int, default=7,
                   help='relu_fourier bands 2^0 through 2^(N-1); ignored by SIREN.')
    p.add_argument('--intensity_head', choices=('sigmoid', 'linear'), default='sigmoid',
                   help='linear: relu_fourier raw intensity, no sigmoid/exp or training clamp.')
    p.add_argument('--time_mode', choices=('local', 'feature', 'alpha'), default='local',
                   help='local: clip-normalized t; feature: no decoder t; alpha: original interval phase.')
    p.add_argument('--loss', choices=('charbonnier', 'lpips'), default='charbonnier')
    p.add_argument('--voxel_normalization', choices=('auto', 'none', 'nonzero'), default='auto')
    p.add_argument('--no_augmentation', action='store_true',
                   help='Disable augmentation for both NPY and H5 datasets and use fixed center crops.')
    p.add_argument('--best_metric', choices=('psnr', 'ssim'), default='psnr')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for name, default in dict(steps=40, crop=112, epochs=50, batch_size=1, workers=4,
                             seed=42, temporal_start=2).items():
        p.add_argument('--' + name, type=int, default=default)
    for name, default in dict(charbonnier_eps=1e-3, lr=1e-4, temporal_weight=0.).items():
        p.add_argument('--' + name, type=float, default=default)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
