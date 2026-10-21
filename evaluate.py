"""Evaluate checkpoints, restoring query coordinates, Fourier bands and intensity head."""

import csv
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


HQF_SEQUENCES = (
    'bike_bay_hdr', 'boxes', 'desk', 'desk_fast', 'desk_hand_only', 'desk_slow',
    'engineering_posters', 'high_texture_plants', 'poster_pillar_1',
    'poster_pillar_2', 'reflective_materials', 'slow_and_fast_desk',
    'slow_hand', 'still_life',
)

CUSTOM_SEQUENCES = ('E2V_custom1', 'E2V_custom2', 'E2V_custom3')
CUSTOM_START_SECONDS = 5.0
MEMMAP_FILES = {
    'events_ts.npy', 'events_xy.npy', 'events_p.npy',
    'images.npy', 'images_ts.npy', 'image_event_indices.npy',
}
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)  # Tile counts, not tile dimensions in pixels.

if __package__:
    from .data import ECDSequence, Sequence, normalize_voxels, read_manifest
    from .train import evaluate_sequence, load_checkpoint, stream_chunks, stream_sequence, save_preview
else:
    from data import ECDSequence, Sequence, normalize_voxels, read_manifest
    from train import evaluate_sequence, load_checkpoint, stream_chunks, stream_sequence, save_preview


def reconstruction_image(log_image, mode='none', gamma=0.6, intensity_head='sigmoid'):
    if intensity_head == 'linear':
        if mode != 'none':
            raise ValueError('Linear intensity requires tone_mapping=none; legacy log tone mapping does not apply.')
        return log_image.clamp(0, 1)
    if intensity_head != 'sigmoid':
        raise ValueError(f'Unknown intensity head: {intensity_head}')
    if mode == 'none':
        return log_image.clamp(max=0).exp()
    if mode == 'evinr_code':
        if gamma <= 0:
            raise ValueError('tone_gamma must be positive.')
        return log_image.sigmoid().pow(1 / gamma)
    raise ValueError(f'Unknown tone-mapping mode: {mode}')


def apply_clahe(images):
    """EVREAL recipe: clamp -> img_as_ubyte -> OpenCV CLAHE -> float32."""
    import cv2
    from skimage.util import img_as_ubyte, img_as_float32
    images = np.asarray(images)
    if images.ndim != 3 or min(images.shape) < 1 or not np.isfinite(images).all():
        raise ValueError('CLAHE expects finite nonempty grayscale [T,H,W] images.')
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return np.stack([img_as_float32(clahe.apply(img_as_ubyte(np.clip(image, 0, 1))))
                     for image in images])


@torch.no_grad()
def evaluation_images(outputs, target, mode='none', gamma=0.6, intensity_head='sigmoid'):
    mapping = 'none' if mode == 'clahe' else mode
    prediction = reconstruction_image(outputs, mapping, gamma, intensity_head).cpu().numpy()
    if mode == 'clahe':
        return apply_clahe(prediction), apply_clahe(target)
    return prediction, target


def clahe_settings():
    import cv2
    return dict(histogram_equalization='clahe', backend='OpenCV', opencv_version=cv2.__version__,
                clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=list(CLAHE_TILE_GRID_SIZE),
                applied_to='prediction and GT independently per frame',
                conversion='clamp_0_1 -> skimage.img_as_ubyte -> CLAHE -> skimage.img_as_float32',
                metric_inputs='same processed frames as primary reconstruction/GT previews',
                settings_source='EVREAL public code; exact EvINR parameters are not published')


def record_clahe(report):
    settings = clahe_settings()
    report['metric_preprocessing'] = settings
    if 'metric_settings' in report:
        report['metric_settings'] = {**report['metric_settings'],
                                     'protocol': report['metric_settings'].get('protocol', 'evreal_std') + '+opencv_clahe',
                                     'histogram_equalization': 'clahe', 'clahe': settings}


@torch.no_grad()
def evaluate_processed_sequence(model, sequence, config, device, preview, mode, gamma):
    if __package__:
        from .eval_utils import ImageMetrics
    else:
        from eval_utils import ImageMetrics
    scorer = ImageMetrics(device=device)  # No internal equalization or additional LPIPS metric.
    model.eval()
    rows = []
    for first, _, _, target, outputs in stream_sequence(
            model, sequence, config.steps, device, config.voxel_normalization):
        if not torch.isfinite(outputs).all():
            raise FloatingPointError('Non-finite decoder output.')
        prediction, gt = evaluation_images(outputs, target[1:], mode, gamma, model.intensity_head)
        scores = scorer.sequence(prediction, gt)
        scores['brightness_bias'] = float((prediction.astype(np.float64) - gt).mean())
        rows.append(dict(scores, frames=len(prediction)))
        if first == 0 and preview is not None:
            save_preview(gt, prediction, preview)
    count = sum(row['frames'] for row in rows)
    return dict({key: sum(row[key] * row['frames'] for row in rows) / count
                 for key in ('l1', 'mse', 'psnr_db', 'ssim', 'brightness_bias')}, frames=count)


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_ecoco(args):
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=False)
    model, saved = load_checkpoint(args.checkpoint, args.device)
    config = Namespace(**saved['args'])
    mode, gamma = getattr(args, 'tone_mapping', 'none'), getattr(args, 'tone_gamma', .6)
    rows = {}

    for path in tqdm(read_manifest(args.data), desc='Evaluate', mininterval=10.):
        sequence = Sequence(path)
        preview = destination / f'{path.name}.png'
        if mode == 'none':
            rows[path.name] = evaluate_sequence(
                model, sequence, config.steps, args.device, preview, config.voxel_normalization)
        else:
            rows[path.name] = evaluate_processed_sequence(
                model, sequence, config, args.device, preview, mode, gamma)

    frames = sum(row['frames'] for row in rows.values())
    mean = {key: sum(row[key] * row['frames'] for row in rows.values()) / frames
            for key in ('l1', 'mse', 'psnr_db', 'ssim', 'brightness_bias')}

    report = dict(
        checkpoint=args.checkpoint, epoch=saved['epoch'], args=saved['args'], mean=mean, sequences=rows,
        protocol=(
            'ecoco frames1..N; persistent causal recurrent stream; '
            'raw intensity metrics; preview-only clamp and p1/p99 normalization'
        ),
        temporal_support='non-overlapping chunks; each voxel encoded once; short tail retained',
        input_normalization=config.voxel_normalization,
    )
    if mode != 'none':
        report.update(tone_mapping=mode,
                      tone_gamma=gamma if mode == 'evinr_code' else None,
                      protocol='ecoco frames1..N; persistent causal recurrent stream; processed intensity metrics')
    if mode == 'clahe':
        record_clahe(report)
    metrics_path = destination / 'metrics.json'
    metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(mean, indent=2))


@torch.inference_mode()
def evaluate_ecd_sequence(model, sequence, config, scorer, device, cut, destination,
                          metric_batch_size=16, video_fps=25, save_images=False,
                          tone_mapping='none', tone_gamma=0.6):
    if __package__:
        from .eval_utils import EvaluationVideos
    else:
        from eval_utils import EvaluationVideos

    default_start, default_end = getattr(
        sequence, 'recording_range', (sequence.times[0], sequence.times[-1]))
    start = float(cut.get('start_time_s', default_start))
    end = float(cut.get('end_time_s', default_end))
    if not np.isfinite([start, end]).all() or start > end:
        raise ValueError('Evaluation cuts require finite start <= end in source-clock seconds.')
    eligible = np.flatnonzero((sequence.times >= start) & (sequence.times <= end))
    if not len(eligible):
        raise ValueError(f'{sequence.path}: no GT in [{start}, {end}]; use the converted common seconds clock.')
    eligible_set = set(map(int, eligible))

    # Match EVREAL: infer at most 10 seconds before the scored interval.
    warmup_time = max(float(sequence.times[0]), start - 10.)
    first = min(int(np.searchsorted(sequence.times, warmup_time, side='left')), int(eligible[0]))
    stop = int(eligible[-1]) + 1

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if save_images:
        (destination / 'recon').mkdir()

    # Keep one encoded frame per source sample so video_index.csv is exact.
    videos = EvaluationVideos(
        destination, video_fps, sample_aligned=True
    ) if video_fps else None
    predictions, references, scored = [], [], []
    video_index = []
    biases = []
    contexts, states = 0, None
    model.eval()
    time_origin = torch.as_tensor([float(sequence.times[0])], device=device, dtype=torch.float64)

    try:
        chunks = stream_chunks(first, stop, config.steps)
        progress = tqdm(chunks, desc=f'Recon {sequence.path.name}', mininterval=10.,
                        miniters=1, unit='chunk', leave=False)

        for begin, count in progress:
            voxel, clock, target = sequence.read_packets(begin, count)
            values = normalize_voxels(voxel, config.voxel_normalization)
            times = torch.as_tensor(clock, device=device)[None]
            context, states = model.encode(torch.as_tensor(values, device=device)[None], times, states, time_origin=time_origin)
            log_tensor = model.render(context, times)[0]
            if not torch.isfinite(log_tensor).all():
                raise FloatingPointError(f'{sequence.path}: nonfinite decoder output.')

            prediction, target = evaluation_images(log_tensor, target, tone_mapping, tone_gamma,
                                                    model.intensity_head)
            contexts += 1

            for local, index in enumerate(range(begin, begin + count)):
                timestamp = float(sequence.times[index])
                image, gt = prediction[local], target[local]

                if videos:
                    signed = voxel[local].sum(0)
                    videos.push(signed, image, timestamp, gt)
                    video_index.append(dict(
                        video_frame=len(video_index),
                        source_frame_index=index,
                        timestamp_s=timestamp,
                        scored=index in eligible_set,
                    ))

                if index not in eligible_set:
                    continue

                predictions.append(image)
                references.append(gt)
                biases.append(float((image.astype(np.float64) - gt).mean()))
                scored.append(index)
                if save_images:
                    Image.fromarray((image * 255).round().astype(np.uint8)).save(
                        destination / 'recon' / f'{index:010d}.png')
    finally:
        video_info = videos.close() if videos else None

    if video_info and video_info['frames'] != len(video_index):
        raise RuntimeError(
            f'Encoded video/index mismatch: {video_info["frames"]} frames, '
            f'{len(video_index)} index rows.')

    video_index_file = None
    scored_video_frames = [row['video_frame'] for row in video_index if row['scored']]
    if video_index:
        video_index_file = 'video_index.csv'
        write_csv(destination / video_index_file, video_index, video_index[0])

    print(f'Metrics {sequence.path.name}: {len(scored)} frames, batch={metric_batch_size}', flush=True)
    mean = scorer.sequence(predictions, references, batch_size=metric_batch_size)
    mean['brightness_bias'] = float(np.mean(biases))

    report = dict(
        method='Ours', data_path=str(sequence.path.resolve()), frames=len(scored), mean=mean,
        metric_settings=scorer.settings(), score_cut_seconds=[start, end],
        scored_time_range_seconds=[float(sequence.times[scored[0]]), float(sequence.times[scored[-1]])],
        first_scored_index=scored[0], last_scored_index=scored[-1],
        steps=config.steps, contexts=contexts, videos=video_info,
        input_normalization=f'{config.voxel_normalization} (matches training)',
        intensity_head=model.intensity_head,
        coord_dim=model.coord_dim, fourier_bands=model.fourier_bands,
        prediction_postprocess=(
            'clamp(raw_intensity,0,1)' if model.intensity_head == 'linear'
            else 'exp(min(L,0))' if tone_mapping in ('none', 'clahe')
            else f'sigmoid(L) ** (1/{tone_gamma:g})'
        ),
        tone_mapping=tone_mapping,
        tone_gamma=tone_gamma if tone_mapping == 'evinr_code' else None,
        protocol=(
            'EVREAL std stored packet boundaries, five bins; packet/frame 0..N-2'
        ),
        conditioning=(
            'persistent recurrent state; every packet encoded once in chronological order'
        ),
        temporal_support='frame0 comes from EVREAL empty packet0; all following frames are causal',
        continuous_query_scope=(
            f'actual-clock feature interpolation; decoder time_mode={model.time_mode}; '
            'benchmark renders endpoints only'
        ),
        evaluation_windows='non-overlapping chunks with persistent state and a short tail',
        evreal_packet_protocol=True, max_lookahead_s=0.,
        conditioning_start_s=float(sequence.times[first]),
        conditioning_end_s=float(sequence.times[stop - 1]),
        video_frames=len(video_index),
        video_source_index_range=(
            [video_index[0]['source_frame_index'], video_index[-1]['source_frame_index']]
            if video_index else None
        ),
        video_time_range_seconds=(
            [video_index[0]['timestamp_s'], video_index[-1]['timestamp_s']]
            if video_index else None
        ),
        score_video_frame_range=(
            [scored_video_frames[0], scored_video_frames[-1] + 1]
            if scored_video_frames else None
        ),
        video_index_file=video_index_file,
        video_frame_range_convention='score_video_frame_range is [start, end), zero-based',
        video_range_policy=(
            'videos contain every reconstructed frame from conditioning_start_s '
            'through the last scored frame; video_index.csv marks scored frames'
        ),
        context_cut_policy=(
            'causal warmup begins at max(sequence start, score start - 10 s); '
            'conditioning stops at the last scored frame'
        ),
        metric_execution='batched after sequence reconstruction; mean per-frame metrics',
        per_frame_metrics_saved=False, metric_batch_size=metric_batch_size,
    )
    if tone_mapping == 'clahe':
        record_clahe(report)
        report['prediction_postprocess'] += ' -> OpenCV CLAHE'
    metrics_path = destination / 'metrics.json'
    metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def dataset_plan(root, data_root, dataset, selected=None):
    """Read official ECD/HQF/MVSEC cuts; Custom scores from source-clock 5s."""
    root, data_root = Path(root), Path(data_root)
    if dataset not in ('ECD', 'HQF', 'MVSEC', 'Custom'):
        raise ValueError('Dataset must be ECD, HQF, MVSEC, or Custom.')
    if not data_root.is_dir():
        raise FileNotFoundError(f'Missing dataset directory: {data_root}')

    if dataset == 'Custom':
        specs = {name: {} for name in CUSTOM_SEQUENCES}
        configuration = dict(
            root_path=str(data_root.resolve()), get_all_sequences=False,
            has_subfolders=False, sequences=specs, dataset_kwargs={},
            score_start_seconds=CUSTOM_START_SECONDS, score_start_reference='source_clock')
    else:
        configuration = json.loads(
            (root / 'config/dataset' / f'{dataset}.json').read_text(encoding='utf-8'))
        if configuration.get('dataset_kwargs'):
            raise ValueError('Per-dataset overrides are unsupported; use EVREAL std settings.')
        if configuration.get('get_all_sequences', False):
            nested = configuration.get('has_subfolders', False)
            entries = [(f'{path.parent.name}_{path.name}' if nested else path.name,
                        dict(sequence_path=str(path.resolve())))
                       for path in sorted(data_root.glob('*/*' if nested else '*')) if path.is_dir()]
            if len({name for name, _ in entries}) != len(entries):
                raise ValueError('Discovered sequence names collide.')
            specs = dict(entries)
        else:
            specs = configuration.get('sequences', {})

    if dataset == 'HQF' and selected is None:
        missing = [name for name in HQF_SEQUENCES if name not in specs]
        if missing:
            raise FileNotFoundError(
                'Incomplete official HQF dataset; missing sequence folders: ' + ', '.join(missing))
        names = list(HQF_SEQUENCES)
    else:
        names = list(specs) if selected is None else selected
    if not names or len(names) != len(set(names)) or any(name not in specs for name in names):
        raise ValueError(f'Choose unique {dataset} sequence names from {list(specs)}.')

    plan = []
    for name in names:
        cut = specs[name]
        source = root / cut['sequence_path'] if 'sequence_path' in cut else data_root / name
        if not source.is_dir():
            raise FileNotFoundError(f'Missing {dataset} sequence: {source}')
        if dataset == 'Custom':
            missing = sorted(MEMMAP_FILES.difference(path.name for path in source.iterdir()))
            if missing:
                raise FileNotFoundError(
                    f'Incomplete Custom memmap sequence {source}; missing: {", ".join(missing)}')
            times = np.load(source / 'images_ts.npy', allow_pickle=False).reshape(-1).astype(np.float64)
            if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f'{source}: Custom requires finite increasing GT timestamps.')
            cut['start_time_s'] = CUSTOM_START_SECONDS
            if times[-2] < cut['start_time_s']:
                raise ValueError(f'{source}: no scored GT remains at/after Custom start {CUSTOM_START_SECONDS:g}s '
                                 '(source clock; EVREAL omits the final GT frame).')
        plan.append((name, source, cut))
    return plan, configuration


def evaluate_ecd(args):
    """Evaluate ECD, HQF, MVSEC, or Custom; keep the historical caller name."""
    if __package__:
        from .eval_utils import ImageMetrics
    else:
        from eval_utils import ImageMetrics

    if not args.out or not str(args.out).strip():
        raise ValueError('--out must be a non-empty path.')

    root = Path(args.evreal_root).expanduser().resolve()
    output_root = Path(args.out).expanduser().resolve()
    dataset = getattr(args, 'dataset', 'ECD')
    out = output_root / dataset
    standard = json.loads((root / 'config/eval/std.json').read_text(encoding='utf-8'))
    plan, dataset_config = dataset_plan(root, args.data_root, dataset, args.sequences)

    expected_dataset = dict(num_bins=5, voxel_method={'method': 'between_frames'}, keep_ratio=1.0)
    if (standard['dataset_kwargs'] != expected_dataset or standard['histeq'] != 'none'
            or standard.get('eval_infer_all', False)):
        raise ValueError(
            'Expected EVREAL std: five bins, between_frames, keep_ratio=1, '
            'histeq=none, eval_infer_all=false.')

    names = [name for name, _, _ in plan]
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')

    model, saved = load_checkpoint(args.checkpoint, args.device)
    config = Namespace(**saved['args'])
    scorer = ImageMetrics(lpips=True, device=args.device)
    out.mkdir(parents=True)

    reports, rows = [], []
    print(f"Checkpoint epoch={saved['epoch']}; encoder={config.encoder}; "
          f"coord_dim={model.coord_dim}; fourier_bands={model.fourier_bands}; "
          f"intensity_head={model.intensity_head}", flush=True)

    for name, source, cut in tqdm(plan, desc=f'Ours / {dataset}', mininterval=10., miniters=1):
        sequence = ECDSequence(source)
        try:
            report = evaluate_ecd_sequence(
                model, sequence, config, scorer, args.device, cut, out / name / 'Ours',
                metric_batch_size=args.metric_batch_size,
                video_fps=0 if args.no_videos else args.video_fps, save_images=args.save_images,
                tone_mapping=args.tone_mapping, tone_gamma=args.tone_gamma)
        finally:
            sequence.close()

        reports.append(report)
        rows.append(dict(sequence=name, frames=report['frames'], **report['mean']))
        tqdm.write(f"{name}: PSNR={report['mean']['psnr_db']:.3f}, "
                   f"SSIM={report['mean']['ssim']:.4f}, LPIPS={report['mean']['lpips']:.4f}")

    frames = sum(report['frames'] for report in reports)
    mean = {key: sum(report['mean'][key] * report['frames'] for report in reports) / frames
            for key in ('l1', 'mse', 'psnr_db', 'ssim', 'lpips', 'brightness_bias')}

    rows.append(dict(sequence='ALL (frame-weighted)', frames=frames, **mean))
    write_csv(out / 'summary.csv', rows, rows[-1])

    report = dict(
        checkpoint=str(Path(args.checkpoint).resolve()), epoch=saved['epoch'], args=saved['args'],
        dataset=dataset, selected_sequences=names, frames=frames, mean=mean, sequences=reports,
        dataset_config=dataset_config, subset_requested=args.sequences is not None,
        dataset_scope=('selected_sequences' if args.sequences is not None else
                       'all_discovered_sequences' if dataset_config.get('get_all_sequences') else
                       'all_configured_sequences'),
        metric_settings=scorer.settings(), evreal_standard_config=standard,
        aggregation='mean per-frame scores, frame-weighted across sequences',
        evreal_packet_protocol=True,
        metric_backend='local implementation with settings recorded in metric_settings',
    )
    if args.tone_mapping == 'clahe':
        record_clahe(report)
    benchmark_path = out / 'benchmark.json'
    benchmark_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(mean, indent=2))
    return report


def main(args):
    return evaluate_ecoco(args) if args.data else evaluate_ecd(args)


def parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)

    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument('--data', help='ecoco manifest evaluation (L1/MSE/PSNR/SSIM).')
    data.add_argument('--data_root',
                      help='ECD, HQF, MVSEC, or Custom memmap directory containing sequence folders; enables LPIPS.')
    p.add_argument('--dataset', choices=('ECD', 'HQF', 'MVSEC', 'Custom'), default='ECD',
                   help='Dataset protocol for --data_root (default: ECD).')
    p.add_argument('--evreal_root', default='../EVREAL',
                   help='Read the dataset config and std.json; no EVREAL Python imports.')
    p.add_argument('--sequences', nargs='+',
                   help='Optional subset; Custom defaults to E2V_custom1/2/3.')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--metric_batch_size', type=int, default=16)
    p.add_argument('--video_fps', type=float, default=25)
    p.add_argument('--no_videos', action='store_true')
    p.add_argument('--save_images', action='store_true',
                   help='Also save reconstruction PNGs under recon/.')

    p.add_argument('--tone_mapping', choices=('none', 'evinr_code', 'clahe'), default='none',
                   help='clahe: native intensity + EVREAL OpenCV CLAHE on prediction AND GT.')
    p.add_argument('--tone_gamma', type=float, default=0.6)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
