#!/usr/bin/env python3
"""
train_instantngp_each_frame.py

Improved version to avoid missing transforms.json after running colmap2nerf.py.

What changed (summary):
 - colmap2nerf.py writes its --out path relative to its current working directory unless an absolute path
   is given. Previously we didn't pass --out, so transforms.json was often created in the instant-ngp repo
   root (cwd) and our script looked for it inside the dataset frame folder and dataset root only.
 - Now we add an explicit --out argument (default: <first_frame>/transforms.json) so the output file
   is created where we expect it and can be copied into other folders reliably.
 - We pass --overwrite to colmap2nerf.py to avoid interactive prompts when it needs to recreate data.
 - We also expose --colmap_out to let you override where transforms.json is written.

Usage example:
  python train_instantngp_each_frame.py /path/to/dataset_root \
      --instant_root /path/to/instant-ngp \
      --first_step_n_steps 20000 --following_n_steps 2000 \
      --colmap_aabb_scale 16 --mask_categories 0 1 \
      --colmap_out /path/to/dataset_root/transforms.json \
      --run-args "--mode nerf --lr 0.001"

"""

import argparse
import subprocess
import sys
import os
import json
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_instantngp_each_frame")


def run_subprocess(cmd, cwd=None, env=None):
    logger.info("Running: %s (cwd=%s)", " ".join(map(str, cmd)), cwd or os.getcwd())
    res = subprocess.run(cmd, cwd=cwd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(map(str, cmd))}")
    return res.returncode


def try_int_sorted(paths):
    try:
        return sorted(paths, key=lambda p: int(p.name))
    except Exception:
        return sorted(paths, key=lambda p: p.name)


def main():
    p = argparse.ArgumentParser(description="Train instant-ngp model per-frame using colmap2nerf + run.py")
    p.add_argument('dataset_root', type=Path, help='Root directory with one subdirectory per frame (e.g. 002361)')
    p.add_argument('--instant_root', type=Path, default=Path('.'), help='Path to instant-ngp repo root (contains scripts/colmap2nerf.py and scripts/run.py)')
    p.add_argument('--first_step_n_steps', type=int, default=20000, help='n_steps for the first (full) frame training')
    p.add_argument('--following_n_steps', type=int, default=2000, help='n_steps for subsequent frames')
    p.add_argument('--colmap_aabb_scale', type=int, default=None, help='Optional, forwarded to colmap2nerf.py as --aabb_scale')
    p.add_argument('--mask_categories', type=str, nargs='*', default=None, help='Optional mask categories forwarded to colmap2nerf.py as --mask_categories')
    p.add_argument('--colmap_extra', type=str, default='', help='Extra flags to forward to colmap2nerf.py (as single string)')
    p.add_argument('--dry_run', action='store_true', help='Print commands without running')
    p.add_argument('--overwrite_transforms', action='store_true', help='Overwrite existing transforms.json in the dataset root and frames')
    p.add_argument('--snapshot_name_template', type=str, default='snapshot_{frame}.msgpack', help='Template for snapshot filenames placed inside each frame dir')
    p.add_argument('--python_bin', type=str, default=sys.executable, help='Python executable to use (default: current interpreter)')
    p.add_argument('--run-args', type=str, default='', help='Extra args (single string) forwarded directly to scripts/run.py')
    p.add_argument('--colmap_matcher', type=str, default=None, help='Optional --colmap_matcher value for colmap2nerf.py (e.g. exhaustive)')
    p.add_argument('--colmap_out', type=Path, default=None, help='Where colmap2nerf.py should write transforms.json (default: <first_frame>/transforms.json)')
    p.add_argument('--colmap_overwrite', action='store_true', help='Pass --overwrite to colmap2nerf.py to avoid interactive prompts')

    args = p.parse_args()

    dataset_root: Path = args.dataset_root.expanduser().resolve()
    instant_root: Path = args.instant_root.expanduser().resolve()

    if not dataset_root.is_dir():
        logger.error("dataset_root is not a directory: %s", dataset_root)
        sys.exit(2)

    scripts_dir = instant_root / 'scripts'
    colmap_script = scripts_dir / 'colmap2nerf.py'
    run_script = scripts_dir / 'run.py'

    if not colmap_script.exists():
        logger.error("colmap2nerf.py not found at %s", colmap_script)
        sys.exit(2)
    if not run_script.exists():
        logger.error("run.py not found at %s", run_script)
        sys.exit(2)

    # list immediate subdirectories only
    entries = [d for d in dataset_root.iterdir() if d.is_dir()]
    if not entries:
        logger.error("No subdirectories found in dataset_root: %s", dataset_root)
        sys.exit(2)

    entries_sorted = try_int_sorted(entries)
    first_dir = entries_sorted[0]
    logger.info("First directory (used to run colmap2nerf.py): %s", first_dir)

    # decide where colmap2nerf.py should write transforms.json
    if args.colmap_out is None:
        # default to writing transforms.json into first_dir so it's easy to pick up
        out_path = (first_dir / 'transforms.json').resolve()
    else:
        out_path = args.colmap_out.expanduser().resolve()

    logger.info("Will ask colmap2nerf.py to write transforms.json to: %s", out_path)

    # Build colmap2nerf command
    colmap_cmd = [args.python_bin, str(colmap_script), '--run_colmap', '--overwrite', '--images', str(first_dir), '--out', str(out_path)]
    if args.colmap_aabb_scale is not None:
        colmap_cmd += ['--aabb_scale', str(args.colmap_aabb_scale)]
    if args.mask_categories is not None and len(args.mask_categories) > 0:
        colmap_cmd += ['--mask_categories'] + list(args.mask_categories)
    if args.colmap_matcher:
        colmap_cmd += ['--colmap_matcher', args.colmap_matcher]
    if args.colmap_extra:
        colmap_cmd += args.colmap_extra.split()
    if args.colmap_overwrite:
        colmap_cmd += ['--overwrite']

    if args.dry_run:
        logger.info("DRY RUN: would run: %s", " ".join(colmap_cmd))
    else:
        # run colmap2nerf from the instant_root so that relative paths used by the script behave the same
        run_subprocess(colmap_cmd, cwd=str(instant_root))

    # After running, check the expected out_path exists
    if not out_path.exists():
        # try some common fallbacks where colmap2nerf might have written the file
        candidates = [
            dataset_root / 'transforms.json',
            instant_root / 'transforms.json',
            first_dir / 'transforms.json'
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            logger.error("Could not find transforms.json after running colmap2nerf.py - please inspect output. Checked: %s", ", ".join(str(x) for x in candidates))
            sys.exit(3)
        else:
            logger.warning("transforms.json was not at requested --out (%s) but found at %s. Using that one.", out_path, found)
            out_path = found

    logger.info("Using transforms.json at %s", out_path)

    # Copy transforms.json to dataset root and into every frame folder
    target_dataset_transform = dataset_root / 'transforms.json'
    if target_dataset_transform.exists() and not args.overwrite_transforms:
        logger.info("transforms.json already exists at %s (use --overwrite_transforms to replace)", target_dataset_transform)
    else:
        logger.info("Copying %s -> %s", out_path, target_dataset_transform)
        if not args.dry_run:
            shutil.copy2(out_path, target_dataset_transform)

    for d in entries_sorted:
        dest = d / 'transforms.json'
        if dest.exists() and not args.overwrite_transforms:
            logger.debug("transforms.json already exists in %s", d)
            continue
        logger.info("Copying transforms.json to %s", dest)
        if not args.dry_run:
            shutil.copy2(out_path, dest)

    # Now run training per frame
    previous_snapshot = None
    for idx, frame_dir in enumerate(entries_sorted):
        frame_name = frame_dir.name
        is_first = (idx == 0)
        n_steps = args.first_step_n_steps if is_first else args.following_n_steps

        snapshot_filename = args.snapshot_name_template.format(frame=frame_name)
        snapshot_path = frame_dir / snapshot_filename

        run_cmd = [args.python_bin, str(run_script), '--scene', str(frame_dir), '--n_steps', str(n_steps), '--save_snapshot', str(snapshot_path)]

        if not is_first and previous_snapshot is not None:
            run_cmd += ['--load_snapshot', str(previous_snapshot)]

        if args.run_args:
            run_cmd += args.run_args.split()

        logger.info("Starting training for frame %s: n_steps=%s (first=%s)", frame_name, n_steps, is_first)
        if args.dry_run:
            logger.info("DRY RUN: would run: %s", " ".join(run_cmd))
        else:
            try:
                run_subprocess(run_cmd, cwd=str(instant_root))
            except Exception as e:
                logger.error("Training for frame %s failed: %s", frame_name, e)
                raise

        previous_snapshot = snapshot_path

    logger.info("All frames processed. Last snapshot: %s", previous_snapshot)


if __name__ == '__main__':
    main()
