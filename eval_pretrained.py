"""Evaluate EVREAL pretrained models without writing image files.

The official EVREAL data protocol and model adapters are reused.  Reconstructions,
event visualisations, and GT are streamed directly to browser-compatible H.264
MP4 files.  L1/MSE/PSNR/SSIM/LPIPS are saved per sequence and summarized as CSV.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from eval_utils import ImageMetrics


METRICS = ("l1", "mse", "psnr_db", "ssim", "lpips")
ALL_METHODS = ("E2VID", "E2VID+", "ET-Net", "HyperE2VID")
CUSTOM_SEQUENCES = ("E2V_custom1", "E2V_custom2", "E2V_custom3")
CUSTOM_START_SECONDS = 5.0
HQF_SEQUENCES = (
    'bike_bay_hdr', 'boxes', 'desk', 'desk_fast', 'desk_hand_only', 'desk_slow',
    'engineering_posters', 'high_texture_plants', 'poster_pillar_1',
    'poster_pillar_2', 'reflective_materials', 'slow_and_fast_desk',
    'slow_hand', 'still_life',
)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
HQF_FILES = {
    "events_ts.npy",
    "events_xy.npy",
    "events_p.npy",
    "images.npy",
    "images_ts.npy",
    "image_event_indices.npy",
}


class H264Writer:
    """Stream RGB uint8 frames to H.264/yuv420p; no intermediate PNGs."""

    def __init__(self, path, height, width, fps):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to write H.264 MP4 files.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        self.path = path
        self.height, self.width = height, width
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Unexpected video frame shape {frame.shape}.")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed ({code}): {self.path}")


class Scores:
    def __init__(self, lpips_model, device, lpips_batch=16):
        self.scorer = ImageMetrics(device=device)
        self.scorer.perceptual = lpips_model
        self.totals = dict.fromkeys(METRICS, 0.)
        self.count = 0
        self.lpips_batch = lpips_batch
        self.pred_queue, self.gt_queue = [], []

    def update(self, pred, gt):
        pred = np.clip(np.asarray(pred, dtype=np.float32).squeeze(), 0, 1)
        gt = np.clip(np.asarray(gt, dtype=np.float32).squeeze(), 0, 1)

        self.pred_queue.append(pred)
        self.gt_queue.append(gt)
        if len(self.pred_queue) >= self.lpips_batch:
            self._flush()

    def _flush(self):
        if not self.pred_queue:
            return
        count = len(self.pred_queue)
        mean = self.scorer.sequence(self.pred_queue, self.gt_queue, batch_size=self.lpips_batch)
        for name in METRICS:
            self.totals[name] += mean[name] * count
        self.count += count
        self.pred_queue.clear()
        self.gt_queue.clear()

    def mean(self):
        self._flush()
        if not self.count:
            raise RuntimeError("No frames matched the quantitative-evaluation protocol.")
        return self.count, {name: value / self.count for name, value in self.totals.items()}


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


def clahe_settings():
    import cv2
    return dict(histogram_equalization='clahe', backend='OpenCV', opencv_version=cv2.__version__,
                clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=list(CLAHE_TILE_GRID_SIZE),
                applied_to='prediction and GT independently per frame',
                conversion='clamp_0_1 -> skimage.img_as_ubyte -> CLAHE -> skimage.img_as_float32',
                metric_inputs='same processed frames as primary reconstruction/GT previews',
                settings_source='EVREAL public code; exact EvINR parameters are not published')


def as_gray(image):
    image = np.asarray(image, dtype=np.float32).squeeze()
    if image.ndim == 3:
        image = image.mean(axis=-1)
    return np.clip(image, 0, 1)


def gray_u8(image):
    return np.rint(as_gray(image) * 255).astype(np.uint8)


def events_u8(voxel):
    signed = voxel.detach().float().cpu().squeeze(0).sum(0).numpy()
    scale = np.percentile(np.abs(signed), 99)
    signed = np.clip(signed / max(float(scale), 1e-6), -1, 1)

    frame = np.ones((*signed.shape, 3), dtype=np.float32)
    positive, negative = np.clip(signed, 0, 1), np.clip(-signed, 0, 1)
    frame[..., 0] -= negative
    frame[..., 1] -= positive + negative
    frame[..., 2] -= positive
    return np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8)


def open_writers(folder, image, fps):
    height, width = image.shape[:2]
    return {
        "recon": H264Writer(folder / "recon.mp4", height, width, fps),
        "events": H264Writer(folder / "events.mp4", height, width, fps),
        "gt": H264Writer(folder / "gt.mp4", height, width, fps),
    }


def close_writers(writers):
    for writer in writers.values():
        writer.close()


def evaluate_sequence(evreal, dataset_name, sequence, method_name, model,
                      method_config, lpips_model, device, output_root, fps,
                      tone_mapping='none'):
    if tone_mapping not in ('none', 'clahe'):
        raise ValueError('tone_mapping must be none or clahe')
    loader = sequence["data_loader"]
    cropper = evreal.get_cropper(model, loader)
    start, end = float(sequence["start_time_s"]), float(sequence["end_time_s"])
    tolerance = 1e-3
    normalize_events = method_config.get("event_tensor_normalization", False)
    postprocess = method_config.get("post_process_norm", "none")
    folder = output_root / dataset_name / sequence["name"] / method_name
    folder.mkdir(parents=True, exist_ok=False)

    scores = Scores(lpips_model, device)
    writers = None
    video_rows = []
    frame_indices = getattr(loader.dataset, "frames_to_use", None)
    model.reset_states()

    try:
        progress = tqdm(loader, desc=f"{method_name} / {dataset_name} / {sequence['name']}")
        for source_index, item in enumerate(progress):
            pred_time = float(item["voxel_timestamp"].item())
            if pred_time < start - 10:
                continue
            if pred_time > end:
                break

            raw_voxel = item["events"]
            voxel = evreal.normalize_event_tensor(raw_voxel) if normalize_events else raw_voxel
            voxel = cropper.pad(voxel.to(device))

            with torch.inference_mode():
                prediction = model(voxel)["image"]
            prediction = evreal.torch2cv2(cropper.crop(prediction))
            prediction = as_gray(evreal.post_process_normalization(prediction, postprocess))
            reference = as_gray(evreal.torch2cv2(item["frame"]))
            if tone_mapping == 'clahe':
                prediction, reference = apply_clahe(np.stack([prediction, reference]))

            if writers is None:
                writers = open_writers(folder, prediction, fps)
            writers["recon"].write(gray_u8(prediction))
            writers["events"].write(events_u8(raw_voxel))
            writers["gt"].write(gray_u8(reference))

            ref_time = float(item["frame_timestamp"].item())
            if abs(pred_time - ref_time) > 1e-9:
                raise RuntimeError(
                    "The std between_frames protocol requires identical prediction "
                    f"and reference timestamps, got {pred_time} and {ref_time}.")
            frame_index = int(frame_indices[source_index]) if frame_indices is not None else source_index
            scored = start <= pred_time <= end and abs(pred_time - ref_time) <= tolerance
            video_rows.append({
                "video_frame": len(video_rows),
                "source_frame_index": frame_index,
                "timestamp_s": ref_time,
                "prediction_timestamp_s": pred_time,
                "reference_timestamp_s": ref_time,
                "scored": int(scored),
            })
            if scored:
                scores.update(prediction, reference)
    finally:
        if writers:
            close_writers(writers)

    frames, mean = scores.mean()
    index_path = folder / "video_index.csv"
    write_csv_fields(
        index_path, video_rows,
        ("video_frame", "source_frame_index", "timestamp_s",
         "prediction_timestamp_s", "reference_timestamp_s", "scored"),
    )
    scored_video_frames = [row["video_frame"] for row in video_rows if row["scored"]]

    report = {
        "dataset": dataset_name,
        "sequence": sequence["name"],
        "method": method_name,
        "frames": frames,
        "video_frames": len(video_rows),
        "video_index_file": index_path.name,
        "video_timestamp_semantics": (
            "timestamp_s is the paired reference-frame time; under EVREAL std "
            "between_frames it is asserted equal to voxel/prediction time"
        ),
        "score_video_frame_range": [
            min(scored_video_frames), max(scored_video_frames) + 1,
        ],
        "score_cut_seconds": [start, end],
        "mean": mean,
        "tone_mapping": tone_mapping,
        "event_tensor_normalization": normalize_events,
        "reconstruction_postprocess": postprocess,
        "metric_settings": scores.scorer.settings(),
        "videos": {name: f"{name}.mp4" for name in writers},
    }
    if tone_mapping == 'clahe':
        settings = clahe_settings()
        report['metric_preprocessing'] = settings
        report['metric_settings'].update(protocol='evreal_std+opencv_clahe',
                                         histogram_equalization='clahe', clahe=settings)
    (folder / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def summary_rows(reports):
    rows = []
    for report in reports:
        rows.append({
            "dataset": report["dataset"], "sequence": report["sequence"],
            "method": report["method"], "frames": report["frames"],
            **report["mean"],
        })

    keys = sorted({(row["dataset"], row["method"]) for row in rows})
    for dataset, method in keys:
        group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
        weights = np.asarray([row["frames"] for row in group], dtype=np.float64)
        rows.append({
            "dataset": dataset, "sequence": "ALL (frame-weighted)", "method": method,
            "frames": int(weights.sum()),
            **{metric: float(np.average([row[metric] for row in group], weights=weights))
               for metric in METRICS},
        })
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("dataset", "sequence", "method", "frames", *METRICS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_fields(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_overall(rows):
    overall = [row for row in rows if row["sequence"] == "ALL (frame-weighted)"]
    print("\nOverall metrics (frame-weighted across sequences)")
    print("dataset  method          frames      L1       MSE    PSNR    SSIM   LPIPS")
    for row in overall:
        print(f"{row['dataset']:<8} {row['method']:<14} {row['frames']:>6}  "
              f"{row['l1']:.5f}  {row['mse']:.5f}  {row['psnr_db']:.3f}  "
              f"{row['ssim']:.4f}  {row['lpips']:.4f}")


def load_evreal(root, device):
    root = Path(root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import eval as evreal
    evreal.device = device
    return evreal


def converted_hqf_sequences(root):
    """Return converted HQF memmap folders and ignore raw bag files."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HQF root does not exist: {root}")

    sequences = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        names = {file.name for file in path.iterdir() if file.is_file()}
        if HQF_FILES.issubset(names):
            sequences.append(path)
    return sequences


def make_hqf_view(root):
    available = {sequence.name: sequence for sequence in converted_hqf_sequences(root)}
    missing = sorted(set(HQF_SEQUENCES) - available.keys())
    if missing:
        raise RuntimeError(
            f"HQF requires all 14 official converted sequences. Missing in {root}: "
            + ', '.join(missing)
        )
    sequences = [available[name] for name in HQF_SEQUENCES]

    temporary = tempfile.TemporaryDirectory(prefix="eventinr_hqf_")
    view = Path(temporary.name)
    for sequence in sequences:
        (view / sequence.name).symlink_to(sequence, target_is_directory=True)

    print(f"HQF: using {len(sequences)} converted sequences from {root}")
    return temporary, view


def custom_config(root):
    """Custom: infer before the cut, but score from source-clock 5.0 seconds."""
    root = Path(root).expanduser().resolve()
    problems = []
    sequences = {}

    for name in CUSTOM_SEQUENCES:
        folder = root / name
        if not folder.is_dir():
            problems.append(f"{folder}: missing directory")
            continue

        names = {path.name for path in folder.iterdir() if path.is_file()}
        missing = sorted(HQF_FILES - names)
        if missing:
            problems.append(f"{folder}: missing {', '.join(missing)}")
            continue
        times = np.load(folder / 'images_ts.npy', allow_pickle=False).reshape(-1).astype(np.float64)
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f'{folder}: Custom requires finite increasing GT timestamps.')
        start = CUSTOM_START_SECONDS
        if times[-2] < start:
            raise ValueError(f'{folder}: no scored GT remains at/after Custom start {start:g}s '
                             '(source clock; EVREAL omits the final GT frame).')
        sequences[name] = {'start_time_s': start}

    if problems:
        raise RuntimeError(
            "Custom evaluation requires converted EVREAL memmap folders:\n  "
            + "\n  ".join(problems)
        )

    return {
        "name": "Custom",
        "root_path": str(root),
        "sequences": sequences,
        "score_start_seconds": CUSTOM_START_SECONDS,
        "score_start_reference": "source_clock",
    }


def parser():
    command = argparse.ArgumentParser()
    command.add_argument("--evreal_root", required=True)
    command.add_argument("--dataset", required=True, choices=("ECD", "HQF", "MVSEC", "Custom"))
    command.add_argument("--data_root", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    command.add_argument("--device", default="cuda")
    command.add_argument("--fps", type=float, default=25)
    command.add_argument("--tone_mapping", choices=('none', 'clahe'), default='none',
                         help='OpenCV CLAHE on prediction AND GT before metrics and videos.')
    return command


def main(args):
    output_root = Path(args.out).resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    destination = output_root / args.dataset
    if destination.exists():
        raise FileExistsError(f'Choose a new output directory: {destination}')
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('fps must be finite and positive')
    device = torch.device(args.device)
    evreal = load_evreal(args.evreal_root, device)

    import lpips
    lpips_model = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()
    lpips_model.requires_grad_(False)
    eval_config = evreal.get_eval_configs(["std"])[0]
    expected = dict(num_bins=5, voxel_method={'method': 'between_frames'}, keep_ratio=1.)
    if (eval_config['dataset_kwargs'] != expected or eval_config['histeq'] != 'none'
            or eval_config.get('eval_infer_all', False) or eval_config.get('ts_tol_ms', 1) != 1):
        raise ValueError('Expected unmodified EVREAL std: five bins, between_frames, '
                         'keep_ratio=1, histeq=none, eval_infer_all=false, ts_tol_ms=1.')
    dataset_name = args.dataset
    hqf_temporary = None
    if dataset_name == "HQF":
        hqf_temporary, data_root = make_hqf_view(data_root)

    if dataset_name == "Custom":
        config = custom_config(data_root)
    else:
        config = evreal.get_dataset_configs([dataset_name])[0]
        config["root_path"] = str(data_root)
    if config.get('dataset_kwargs'):
        raise ValueError('Dataset-level voxel overrides would change the shared std packet protocol.')
    dataset_config = deepcopy(config)  # EVREAL attaches runtime DataLoaders to config in place.
    sequences = evreal.get_sequences(
        config, dict(eval_config.get("dataset_kwargs", {})))
    destination.mkdir(parents=True, exist_ok=False)

    reports = []
    for method_name in args.methods:
        method_config = evreal.get_method_config(method_name)
        model = evreal.get_model_from_checkpoint_path(
            method_config["model_name"], method_config["model_path"])
        for sequence in sequences:
            report = evaluate_sequence(
                evreal, dataset_name, sequence, method_name, model, method_config,
                lpips_model, device, output_root, args.fps, args.tone_mapping)
            reports.append(report)
            print(f"{dataset_name}/{sequence['name']}/{method_name}: "
                  + ", ".join(f"{key}={value:.5f}" for key, value in report["mean"].items()))

    rows = summary_rows(reports)
    write_csv(output_root / dataset_name / "summary.csv", rows)
    benchmark = dict(dataset=dataset_name, methods=args.methods, tone_mapping=args.tone_mapping,
                     sequences=reports, dataset_config=dataset_config, evreal_standard_config=eval_config,
                     evreal_packet_protocol=True,
                     metric_backend='shared eval_utils.ImageMetrics; standalone LPIPS Alex v0.1, not PyIQA',
                     aggregation='mean per-frame scores, frame-weighted across sequences')
    (destination / 'benchmark.json').write_text(json.dumps(benchmark, indent=2), encoding='utf-8')
    print_overall(rows)
    if hqf_temporary:
        hqf_temporary.cleanup()


if __name__ == "__main__":
    main(parser().parse_args())
