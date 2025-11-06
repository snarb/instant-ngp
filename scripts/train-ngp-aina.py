#!/usr/bin/env python3
"""
Instant-NGP per-frame training and rendering automation.

This script automates:
1. Running COLMAP once on a reference frame to generate transforms.json.
2. Training each frame sequentially, saving per-frame snapshots.
3. Rendering a combined video sequence using the trained snapshots.

Features:
- Recreates logs on each run.
- Stores snapshots per-frame in a specified root.
- Optional rendering with configurable AABB and ffmpeg path.
- Can preserve or clean temporary render directories.

Author: Pavlo Konovalov
Refactored by ChatGPT (GPT-5)
"""

from __future__ import annotations
import sys
import argparse
import shutil
import subprocess
import json
from pathlib import Path
import os
import logging
import time

# Path to pyngp build for side-effect import
pyngp_path = '/home/ubuntu/repos/instant-ngp/build/'
sys.path.append(pyngp_path)
try:
    import pyngp as ngp  # noqa: F401 (side-effect import)
except Exception:
    pass


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure logger to both file and stdout; overwrite existing log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode="w")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Logging initialized. Writing to %s", log_path)
    return logger


def run(cmd, cwd: Path | None = None, env: dict | None = None) -> int:
    """Run a subprocess with streaming logs and fail fast if non-zero exit code."""
    logger = logging.getLogger()
    logger.info("Running: %s", " ".join(map(str, cmd)))

    env2 = os.environ.copy()
    env2.update(env or {})
    env2.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env2,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            logger.info("[child] %s", line.rstrip())
    except Exception:
        logger.exception("While streaming child output")

    rc = process.wait()
    if rc != 0:
        logger.error("Command failed (rc=%s)", rc)
        sys.exit(1)
    return rc


def try_int_sort(paths):
    """Sort directories numerically by name if possible, else lexicographically."""
    try:
        return sorted(paths, key=lambda p: int(p.name))
    except Exception:
        return sorted(paths, key=lambda p: p.name)


def fix_transforms_paths(json_path: Path, source_dir: Path, overwrite: bool = True):
    """Update `file_path` fields in transforms.json to point to actual frame image paths.

    The function replaces the trailing frame token in file names so that
    each `file_path` resolves to `source_dir/<prefix>.<frame_name><ext>`.
    """
    json_path = Path(json_path)
    source_dir = Path(source_dir)
    frame_name = source_dir.name

    if not json_path.exists():
        raise FileNotFoundError(f"transforms.json not found: {json_path}")

    with json_path.open("r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    changed = False
    for frame in frames:
        if "file_path" not in frame:
            continue
        orig_name = Path(frame["file_path"]).name
        ext = Path(orig_name).suffix
        stem = orig_name[:-len(ext)] if ext else orig_name
        prefix = stem.rsplit('.', 1)[0] if '.' in stem else stem
        new_name = f"{prefix}.{frame_name}{ext}" if prefix else f"{frame_name}{ext}"
        new_fp = str((source_dir / new_name).resolve())
        if frame.get("file_path") != new_fp:
            frame["file_path"] = new_fp
            changed = True

    if overwrite and changed:
        with json_path.open("w") as f:
            json.dump(data, f, indent=2)

    return data


def build_sequence(frames, palindrome=False):
    """Return ordered list of frames, optionally mirrored for palindrome rendering."""
    seq = list(frames)
    if palindrome and len(frames) > 1:
        seq = frames + frames[-2::-1]
    return seq


def run_colmap(instant_root: Path, reference_frame: Path, out_path: Path, aabb_scale: int | None,
               multi_camera: bool, mask_categories: list[str] | None, skip_colmap: bool):
    """Run colmap2nerf.py to generate transforms.json, unless skipped."""
    logger = logging.getLogger()
    text_folder = reference_frame / "sparse"
    text_folder.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    colmap_cmd = [
        sys.executable, str(instant_root / "scripts/colmap2nerf.py"),
        "--colmap_db", str(reference_frame / "database.db"),
        "--text", str(text_folder),
        "--images", str(reference_frame),
        "--out", str(out_path),
        "--run_colmap", "--overwrite",
    ]
    if aabb_scale:
        colmap_cmd += ["--aabb_scale", str(aabb_scale)]
    if multi_camera:
        colmap_cmd += ["--multi_camera"]
    if mask_categories:
        colmap_cmd += ["--mask_categories", *map(str, mask_categories)]

    if not skip_colmap:
        logger.info("Running colmap2nerf.py using reference frame: %s", reference_frame.name)
        run(colmap_cmd, cwd=instant_root)
    else:
        logger.info("Skipping COLMAP. Assuming transforms.json exists at: %s", out_path)

    if not out_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {out_path} after COLMAP stage.")


def copy_transforms_to_all(out_path: Path, dataset_root: Path, all_frames: list[Path]):
    """Copy the generated transforms.json to all frame subfolders for consistency."""
    logger = logging.getLogger()
    target_dataset_transform = dataset_root / "transforms.json"
    shutil.copy2(out_path, target_dataset_transform)
    logger.info("Copied %s -> %s", out_path, target_dataset_transform)

    for frame in all_frames:
        dst = frame / "transforms.json"
        if dst.resolve() != out_path.resolve():
            shutil.copy2(out_path, dst)


def train_frames(instant_root: Path, training_frames: list[Path], first_steps: int, follow_steps: int,
                 snapshots_root: Path, extra_args: list[str]):
    """Train Instant-NGP per-frame; save snapshots to snapshots_root/<frame>/.

    Note on schedule of LR:
    After **20,000 steps (decay_start)**, the base learning rate of **1e-2** begins to decay — every
    **10,000 steps (decay_interval)** it is multiplied by **0.33 (decay_base)**.
    Before reaching 20,000 steps, the original learning rate (1e-2) is used.
    """
    logger = logging.getLogger()
    prev_snapshot: Path | None = None
    for idx, frame in enumerate(training_frames):
        frame_start = time.time()
        logger.info("--- Training frame %s (%d/%d) ---", frame.name, idx + 1, len(training_frames))

        # Ensure transforms.json points to the current frame images
        fix_transforms_paths(frame / "transforms.json", frame, overwrite=True)

        # Create per-frame snapshot directory
        frame_snap_dir = snapshots_root / frame.name
        if frame_snap_dir.exists():
            shutil.rmtree(frame_snap_dir, ignore_errors=True)
        frame_snap_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = frame_snap_dir / f"snapshot_{frame.name}.msgpack"
        steps_to_go = first_steps if idx == 0 else first_steps + follow_steps
        resume_step = 0 if idx == 0 else first_steps

        run_cmd = [
            sys.executable, str(instant_root / "scripts/run-ngp-aina.py"),
            "--scene", str(frame),
            "--save_snapshot", str(snapshot_path),
            "--n_steps", str(steps_to_go),
            "--override_training_step", str(resume_step),
        ]
        if idx > 0 and prev_snapshot:
            run_cmd += ["--load_snapshot", str(prev_snapshot)]
        if extra_args:
            run_cmd += list(extra_args) #prev_snapshot = '/fsx/Test_Prism_new/Walking1/cropped_jpgs/sequence1/seqA1/snapshots/000149/snapshot_000149.msgpack'

        run(run_cmd, cwd=instant_root)
        prev_snapshot = snapshot_path
        logger.info("Finished training for %s in %.2f s", frame.name, time.time() - frame_start)


def resolve_snapshot(frame_snap_dir: Path, frame_name: str) -> Path:
    """Return first existing snapshot file for given frame; raise if missing."""
    for ext in (".msgpack", ".ingp"):
        p = frame_snap_dir / f"snapshot_{frame_name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Snapshot not found for frame '{frame_name}' in {frame_snap_dir}")


def render_combined_video(instant_root: Path, camera_path: Path, sequence_frames: list[Path], fps: int,
                          output_path: Path, snapshots_root: Path, keep_frames: bool, extra_args: list[str],
                          aabb_rendering: list[str], ffmpeg_path: str, keep_temp_dir: bool):
    """Render each trained snapshot into a frame and combine into a final video.

    Creates (or reuses) snapshots_root/tmp_render/ for PNG frames, then stitches
    them with ffmpeg into the final video.
    """
    logger = logging.getLogger()
    tmp_dir = snapshots_root / "tmp_render"

    # Recreate tmp render dir unless explicitly told to keep it
    if tmp_dir.exists() and not keep_temp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total_frames_in_sequence = len(sequence_frames)
    logger.info("Preparing to render %d individual frames to create the final video.", total_frames_in_sequence)

    for i, frame in enumerate(sequence_frames):
        render_start = time.time()
        frame_snap_dir = snapshots_root / frame.name
        snapshot = resolve_snapshot(frame_snap_dir, frame.name)

        temp_output_pattern = tmp_dir / "render_%04d.png"
        video_duration = len(sequence_frames)  # 1s per model
        frames_per_segment = fps
        start_frame = frames_per_segment * i
        end_frame = frames_per_segment * (i + 1)

        render_cmd = [
            sys.executable,
            str(instant_root / "scripts/run-ngp-aina.py"),
            "--scene", str(frame),
            "--load_snapshot", str(snapshot),
            "--video_camera_path", str(camera_path),
            "--video_fps", str(fps),
            "--video_n_seconds", str(video_duration),
            "--video_render_range", str(start_frame), str(end_frame),
            "--video_output", str(temp_output_pattern),
        ]

        if aabb_rendering is not None:
            render_cmd += ["--aabb", *map(str, aabb_rendering)]

        if extra_args:
            render_cmd += list(extra_args)

        logger.info("Rendering frame %d/%d using model from %s", i + 1, total_frames_in_sequence, frame.name)
        run(render_cmd, cwd=instant_root)
        logger.info("Frame %d rendered in %.2f s", i + 1, time.time() - render_start)

    logger.info("Stitching frames into final video: %s", output_path)
    ffmpeg_cmd = [
        ffmpeg_path, "-y",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "render_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    run(ffmpeg_cmd)

    if not keep_frames:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            logger.warning("Could not remove temporary directory %s", tmp_dir)


def parse_args():
    """Define and parse CLI arguments for the script."""
    p = argparse.ArgumentParser(description="Train Instant-NGP per-frame and optionally render combined video.")

    # Core
    p.add_argument("dataset_root", type=Path, help="Root directory with frame subdirectories")
    p.add_argument("--instant_root", type=Path, default=Path.cwd(), help="Path to instant-ngp repo root")
    p.add_argument("--snapshots_root", type=Path, required=True,
                   help="Directory to store per-frame snapshots (subfolder per frame) and tmp_render/")

    # Training
    p.add_argument("--first_step_n_steps", type=int, default=40000)
    p.add_argument("--following_n_steps", type=int, default=15000)
    p.add_argument("--start_frame", type=int, default=None)
    p.add_argument("--end_frame", type=int, default=None)

    # COLMAP
    p.add_argument("--aabb_scale", type=int)
    p.add_argument("--mask_categories", nargs="*")
    p.add_argument("--colmap_out", type=Path, help="Where to write transforms.json (default: first frame folder)")
    p.add_argument("--skip_colmap", action="store_true")
    p.add_argument("--multi_camera", action="store_true",
                   help="Forward to colmap2nerf.py to allow multiple cameras (different resolutions).")

    # Rendering
    p.add_argument("--render_only", action="store_true")
    p.add_argument("--render_after_train", action="store_true")
    p.add_argument("--render_camera_path", type=Path)
    p.add_argument("--render_start_frame", type=int, default=None)
    p.add_argument("--render_end_frame", type=int, default=None)
    p.add_argument("--render_fps", type=int, default=30)
    p.add_argument("--render_palindrome", action="store_true")
    p.add_argument("--render_output", type=Path, default=None)
    p.add_argument("--render_keep_frames", action="store_true")
    p.add_argument("--render_extra_args", nargs="*")

    # Rendering region (instead of hardcoded AABB)
    p.add_argument(
		"--aabb_rendering",
		nargs=6,
		type=str,
		default=None,
		metavar=('minx', 'miny', 'minz', 'maxx', 'maxy', 'maxz'),
		help="AABB values for rendering region."
	)


# ffmpeg binary path override
    p.add_argument("--ffmpeg_path", type=str, default=None,
                   help="Custom path to ffmpeg binary; defaults to 'ffmpeg' in PATH.")

    # Keep tmp_render/ between runs
    p.add_argument("--keep_temp_dir", action="store_true",
                   help="Keep temporary render directory instead of recreating.")

    args, extra = p.parse_known_args()
    return args, extra


def filter_frames(all_frames: list[Path], start_frame: int | None, end_frame: int | None, purpose: str) -> list[Path]:
    """Filter frame directories by numeric name range (inclusive)."""
    logger = logging.getLogger()
    if start_frame is None and end_frame is None:
        return all_frames
    logger.info("Filtering %s frames for range: [%s, %s]", purpose, start_frame, end_frame)
    filtered = []
    for frame in all_frames:
        try:
            num = int(frame.name)
            if start_frame is not None and num < start_frame:
                continue
            if end_frame is not None and num > end_frame:
                continue
            filtered.append(frame)
        except ValueError:
            logger.warning("Skipping non-integer frame name '%s' during %s filter", frame.name, purpose)
    return filtered


def main():
    """Entry point: orchestrates COLMAP (optional), training, and rendering."""
    args, extra_args = parse_args()

    dataset_root = args.dataset_root.resolve()
    instant_root = args.instant_root.resolve()
    snapshots_root = args.snapshots_root.resolve()

    # init logging (overwrites previous log file)
    log_file_path = dataset_root / "training_log.txt"
    logger = configure_logging(log_file_path)
    logger.info("--- Starting new script run ---")

    if not dataset_root.is_dir():
        logger.error("Dataset root does not exist or is not a directory: %s", dataset_root)
        sys.exit(1)

    all_frames = try_int_sort([d for d in dataset_root.iterdir() if d.is_dir()])
    if not all_frames:
        logger.error("No frame subdirectories found in dataset_root")
        sys.exit(1)

    # COLMAP + TRAIN unless render-only
    if not args.render_only:
        training_frames = filter_frames(all_frames, args.start_frame, args.end_frame, "training")
        if not training_frames:
            logger.error("No frames matched the specified --start_frame/--end_frame filter. Nothing to train.")
            sys.exit(1)
        logger.info("Found %d frames to train out of %d total.", len(training_frames), len(all_frames))

        reference_frame = all_frames[0]
        out_path = (args.colmap_out.resolve() if args.colmap_out else (reference_frame / "transforms.json"))
        run_colmap(
            instant_root=instant_root,
            reference_frame=reference_frame,
            out_path=out_path,
            aabb_scale=args.aabb_scale,
            multi_camera=args.multi_camera,
            mask_categories=args.mask_categories,
            skip_colmap=args.skip_colmap,
        )

        copy_transforms_to_all(out_path, dataset_root, all_frames)

        snapshots_root.mkdir(parents=True, exist_ok=True)
        train_frames(
            instant_root=instant_root,
            training_frames=training_frames,
            first_steps=args.first_step_n_steps,
            follow_steps=args.following_n_steps,
            snapshots_root=snapshots_root,
            extra_args=(args.render_extra_args or []),
        )
    else:
        logger.info("--- Render-only mode enabled. Skipping COLMAP and training. ---")

    # Rendering stage
    if args.render_after_train or args.render_only:
        if not args.render_camera_path:
            logger.error("Rendering requested but --render_camera_path is missing.")
            sys.exit(1)
        if not args.render_camera_path.exists():
            logger.error("Camera path file not found: %s", args.render_camera_path)
            sys.exit(1)

        rendering_frames = filter_frames(all_frames, args.render_start_frame, args.render_end_frame, "render")
        if not rendering_frames:
            logger.error("No frames matched the specified render frame filter. Nothing to render.")
            sys.exit(1)

        logger.info("Found %d frames to render.", len(rendering_frames))
        final_output = args.render_output.resolve() if args.render_output else (dataset_root / "combined.mp4")
        seq_frames = build_sequence(rendering_frames, palindrome=args.render_palindrome)
        logger.info("Rendering a video from %d total frames/models.", len(seq_frames))

        ffmpeg_path = args.ffmpeg_path if args.ffmpeg_path else "ffmpeg"
        render_combined_video(
            instant_root=instant_root,
            camera_path=args.render_camera_path.resolve(),
            sequence_frames=seq_frames,
            fps=args.render_fps,
            output_path=final_output,
            snapshots_root=snapshots_root,
            keep_frames=args.render_keep_frames,
            extra_args=(args.render_extra_args or []),
            aabb_rendering=args.aabb_rendering,
            ffmpeg_path=ffmpeg_path,
            keep_temp_dir=args.keep_temp_dir,
        )
        logger.info("Combined video saved to: %s", final_output)

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
