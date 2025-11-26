#!/usr/bin/env python3
import sys

pyngp_path = '/home/ubuntu/repos/instant-ngp/build/'
sys.path.append(pyngp_path)
import pyngp as ngp  # noqa: F401 (import kept for compatibility, not used directly here)
import argparse
import shutil
import subprocess
import json
from pathlib import Path
import os
import logging
import time

# Use the root logger configured in main()
logger = logging.getLogger()


def run(cmd, cwd=None):
	"""Executes a command and logs its execution and result."""
	logger.info(f"Running: {' '.join(map(str, cmd))}")
	result = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
	if result.returncode != 0:
		error_msg = f"Command failed: {' '.join(map(str, cmd))} (rc={result.returncode})"
		logger.error(error_msg)
		sys.exit(error_msg)


def try_int_sort(paths):
	"""Tries to sort paths by integer name, falling back to string sort."""
	try:
		return sorted(paths, key=lambda p: int(p.name))
	except Exception:
		return sorted(paths, key=lambda p: p.name)


def fix_transforms_paths(json_path: Path, source_dir: Path, overwrite: bool = True):
	"""
	Replace all "file_path" entries in an instant-ngp transforms.json so that they point to files
	inside `source_dir`, and replace the frame-number suffix in the filename to match the
	source_dir name.

	Image filename format expected: <prefix>.<frame_number>.<ext>
	  e.g. A003_A016_0501OD.002361.jpg
	"""
	json_path = Path(json_path)
	source_dir = Path(source_dir)
	frame_name = source_dir.name  # the frame number we want to insert into filenames

	if not json_path.exists():
		raise FileNotFoundError(f"transforms.json not found: {json_path}")

	with json_path.open("r") as f:
		data = json.load(f)

	frames = data.get("frames", [])
	changed = False
	for frame in frames:
		if "file_path" not in frame:
			continue
		orig_fp = frame["file_path"]
		orig_name = Path(orig_fp).name

		ext = Path(orig_name).suffix  # includes leading dot
		stem = orig_name[:-len(ext)] if ext else orig_name

		if '.' in stem:
			prefix, _ = stem.rsplit('.', 1)
		else:
			prefix = stem

		new_name = f"{prefix}.{frame_name}{ext}" if prefix else f"{frame_name}{ext}"
		new_fp = str((source_dir / new_name).resolve())

		if frame.get("file_path") != new_fp:
			frame["file_path"] = new_fp
			changed = True

	if overwrite and changed:
		with json_path.open("w") as f:
			json.dump(data, f, indent=2)

	return data


def build_sequence(frames, palindrome=False, repeats=1):
	"""
	frames: list of Path (in order)
	palindrome: if True produce [A,B,C,B,A]
	repeats: repeat the whole resulting sequence N times
	"""
	seq = list(frames)
	if palindrome and len(frames) > 1:
		seq = frames + frames[-2::-1]
	if repeats > 1:
		seq = seq * repeats
	return seq


def render_combined_video(instant_root: Path, camera_path: Path, sequence_frames, fps: int, seconds_per_segment: float,
						  output_path: Path, tmp_dir: Path, keep_segments: bool, extra_args):
	"""
	For each Path in sequence_frames, expects snapshot at frame / snapshot_{frame.name}.msgpack
	Renders a short video segment for each snapshot and concatenates them into output_path.
	"""
	instant_root = Path(instant_root)
	tmp_dir = Path(tmp_dir)
	os.makedirs(tmp_dir, exist_ok=True)

	segment_paths = []
	for i, frame in enumerate(sequence_frames):
		render_start_time = time.time()
		snapshot = frame / f"snapshot_{frame.name}.msgpack"
		if not snapshot.exists():
			# try alternative: snapshot with .ingp or fallback to any snapshot in dir
			alt = None
			for ext in (".msgpack", ".ingp"):
				t = frame / f"snapshot_{frame.name}{ext}"
				if t.exists():
					alt = t
					break
			# fallback: any file starting with snapshot_
			if not alt:
				for f in frame.iterdir():
					if f.name.startswith("snapshot_"):
						alt = f
						break
			if alt:
				snapshot = alt
			else:
				logger.warning(f"Snapshot not found for frame {frame}. Skipping this frame.")
				continue

		segment_out = tmp_dir / f"segment_{i:04d}.mp4"
		render_cmd = [
			sys.executable,
			str(instant_root / "scripts/run.py"),
			"--scene", str(frame),
			"--load_snapshot", str(snapshot),
			"--video_camera_path", str(camera_path),
			"--video_fps", str(fps),
			"--video_n_seconds", str(seconds_per_segment),
			"--video_output", str(segment_out)
		]
		if extra_args:
			render_cmd += list(extra_args)

		logger.info(f"Rendering segment {i + 1}/{len(sequence_frames)} for frame {frame.name} -> {segment_out}")
		run(render_cmd, cwd=str(instant_root))

		if not segment_out.exists():
			logger.warning(f"Segment not produced for {frame.name}: expected {segment_out}")
			continue

		segment_paths.append(segment_out)
		render_duration = time.time() - render_start_time
		logger.info(f"Segment for frame {frame.name} rendered in {render_duration:.2f} seconds.")

	if not segment_paths:
		logger.warning("No segments were rendered; skipping concatenation.")
		return

	concat_list = tmp_dir / "segments.txt"
	with concat_list.open("w") as f:
		for p in segment_paths:
			f.write(f"file '{str(p.resolve())}'\n")

	logger.info(f"Concatenating {len(segment_paths)} segments into {output_path}")
	ffmpeg_cmd = [
		"ffmpeg", "-y", "-f", "concat", "-safe", "0",
		"-i", str(concat_list),
		"-c", "copy",
		str(output_path)
	]
	run(ffmpeg_cmd, cwd=None)

	if not keep_segments:
		for p in segment_paths:
			try:
				p.unlink()
			except Exception:
				pass
		try:
			concat_list.unlink()
		except Exception:
			pass
		try:
			tmp_dir.rmdir()
		except Exception:
			pass


def main():
	parser = argparse.ArgumentParser(description="Train Instant-NGP model per frame and optionally render combined video.", add_help=True)
	parser.add_argument("dataset_root", type=Path, help="Root directory with frame subdirectories")
	parser.add_argument("--instant_root", type=Path, default=Path.cwd(), help="Path to instant-ngp repo root (default: current dir)")
	parser.add_argument("--first_step_n_steps", type=int, default=20000, help="Training steps for first frame (default: 20000)")
	parser.add_argument("--following_n_steps", type=int, default=10000, help="Training steps for subsequent frames (default: 10000)")
	parser.add_argument("--aabb_scale", type=int, help="aabb_scale arg for colmap2nerf.py")
	parser.add_argument("--mask_categories", nargs="*", help="mask_categories arg for colmap2nerf.py")
	parser.add_argument("--colmap_out", type=Path, help="Where to write transforms.json (default: first frame folder)")
	parser.add_argument("--skip_colmap", action="store_true", help="do not run the colmap (if transforms.json is already present)")

	# Rendering options
	parser.add_argument("--render_after_train", action="store_true",
						help="After finishing training, render combined video from per-frame snapshots")
	parser.add_argument("--render_camera_path", type=Path,
						help="Camera path JSON to pass as --video_camera_path to run.py (required for rendering)")
	parser.add_argument("--render_fps", type=int, default=30, help="Output FPS for video segments (default: 30)")
	parser.add_argument("--render_seconds_per_segment", type=float, default=1.0,
						help="Seconds to render per snapshot segment (default: 1.0)")
	parser.add_argument("--render_palindrome", action="store_true", help="Make sequence palindrome (a-b-c-b-a)")
	parser.add_argument("--render_repeats", type=int, default=1, help="Repeat the full sequence N times (default:1)")
	parser.add_argument("--render_output", type=Path, default=None,
						help="Final combined video output path (default: dataset_root/combined.mp4)")
	parser.add_argument("--render_tmpdir", type=Path, default=None,
						help="Temporary directory for per-segment files (default: dataset_root/tmp_render)")
	parser.add_argument("--render_keep_segments", action="store_true", help="Do not delete per-segment files after concatenation")
	parser.add_argument("--render_extra_args", nargs="*",
						help="Extra args to forward to run.py during rendering (e.g. --network mynet.json)")

	args, extra_args = parser.parse_known_args()

	dataset_root = args.dataset_root.resolve()

	# --- Setup Logging ---
	log_file_path = dataset_root / "training_log.txt"
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
		handlers=[
			logging.FileHandler(log_file_path),
			logging.StreamHandler(sys.stdout)
		]
	)
	logger.info(f"Starting script. Log file will be saved to: {log_file_path}")
	# ---------------------

	instant_root = args.instant_root.resolve()
	if not dataset_root.exists() or not dataset_root.is_dir():
		logger.error(f"dataset_root does not exist or is not a directory: {dataset_root}")
		sys.exit(1)

	frames = [d for d in dataset_root.iterdir() if d.is_dir()]
	frames = try_int_sort(frames)
	if not frames:
		logger.error("No frame subdirectories found in dataset_root")
		sys.exit(1)

	out_path = (args.colmap_out.resolve() if args.colmap_out else (frames[0] / "transforms.json"))
	try:
		os.makedirs(out_path.parent, exist_ok=True)
	except Exception as e:
		logger.warning(f"Could not create parent directory for out_path {out_path.parent}: {e}")

	text_folder = frames[0] / "sparse"
	try:
		os.makedirs(text_folder, exist_ok=True)
	except Exception as e:
		logger.warning(f"Could not create text_folder {text_folder}: {e}")

	colmap_cmd = [
		sys.executable,
		str(instant_root / "scripts/colmap2nerf.py"),
		"--colmap_db", str(frames[0] / "database.db"),
		"--text", str(text_folder),
		"--images", str(frames[0]),
		"--out", str(out_path),
		"--run_colmap",
		"--overwrite"
	]
	if args.aabb_scale:
		colmap_cmd += ["--aabb_scale", str(args.aabb_scale)]
	if args.mask_categories:
		colmap_cmd += ["--mask_categories"] + list(args.mask_categories)

	if not args.skip_colmap:
		logger.info("About to run colmap2nerf.py with the following settings:")
		logger.info(f"  colmap db: {frames[0] / 'database.db'}")
		logger.info(f"  text output folder (passed as --text): {text_folder}")
		logger.info(f"  images folder: {frames[0]}")
		logger.info(f"  out transforms.json: {out_path}")
		run(colmap_cmd, cwd=instant_root)
	else:
		logger.info(f"Skipping colmap run (--skip_colmap). Assuming transforms.json already present at: {out_path}")

	if not out_path.exists():
		logger.error(f"FATAL: Expected transforms.json not found at {out_path}")
		logger.error("Listing contents of frame[0] folder for debugging:")
		try:
			for p in sorted(frames[0].iterdir()):
				logger.error(f"  {p.name}")
		except Exception as e:
			logger.error(f"  (could not list folder): {e}")
		logger.error(f"Listing contents of text_folder (where colmap2nerf.py writes TXT output): {text_folder}")
		try:
			for p in sorted(text_folder.iterdir()):
				logger.error(f"  {p.name}")
		except Exception as e:
			logger.error(f"  (could not list text_folder): {e}")
		sys.exit(1)

	target_dataset_transform = dataset_root / "transforms.json"
	try:
		shutil.copy2(out_path, target_dataset_transform)
		logger.info(f"Copied {out_path} -> {target_dataset_transform}")
	except Exception as e:
		logger.warning(f"Failed to copy transforms.json to dataset_root: {e}")

	for frame in frames:
		dest = frame / "transforms.json"
		try:
			shutil.copy2(out_path, dest)
		except Exception as e:
			logger.warning(f"Failed to copy transforms.json to {dest}: {e}")

	cur_step = 0
	prev_snapshot = None
	for idx, frame in enumerate(frames):
		frame_start_time = time.time()
		logger.info(f"--- Processing frame {frame.name} ({idx + 1}/{len(frames)}) ---")
		frame_transform = frame / "transforms.json"
		try:
			fix_transforms_paths(frame_transform, frame, overwrite=True)
		except Exception as e:
			logger.warning(f"Failed to fix transforms for {frame_transform}: {e}")

		try:
			os.makedirs(frame, exist_ok=True)
		except Exception:
			pass

		snapshot_path = frame / f"snapshot_{frame.name}.msgpack"
		steps_to_go = args.first_step_n_steps if idx == 0 else args.following_n_steps
		cur_step += steps_to_go
		run_cmd = [
			sys.executable,
			str(instant_root / "scripts/run.py"),
			"--scene", str(frame),
			"--save_snapshot", str(snapshot_path),
			"--n_steps", str(cur_step)
		]
		if idx > 0 and prev_snapshot:
			run_cmd += ["--load_snapshot", str(prev_snapshot)]

		if extra_args:
			run_cmd += extra_args

		logger.info(f"Running training for frame: {frame.name}")
		run(run_cmd, cwd=instant_root)
		prev_snapshot = snapshot_path

		frame_duration = time.time() - frame_start_time
		logger.info(f"Finished training for {frame.name}. Frame processing took {frame_duration:.2f} seconds.")

	if args.render_after_train:
		if not args.render_camera_path:
			logger.error("Rendering requested but --render_camera_path is missing.")
			sys.exit(1)

		camera_path = args.render_camera_path.resolve()
		if not camera_path.exists():
			logger.error(f"Camera path file not found: {camera_path}")
			sys.exit(1)

		final_output = args.render_output.resolve() if args.render_output else (dataset_root / "combined.mp4")
		tmp_dir = (args.render_tmpdir.resolve() if args.render_tmpdir else (dataset_root / "tmp_render"))
		os.makedirs(tmp_dir, exist_ok=True)

		seq_frames = build_sequence(frames, palindrome=args.render_palindrome, repeats=max(1, args.render_repeats))
		logger.info(
			f"Rendering combined video: {len(seq_frames)} segments (palindrome={args.render_palindrome}, repeats={args.render_repeats})")
		render_combined_video(
			instant_root=instant_root,
			camera_path=camera_path,
			sequence_frames=seq_frames,
			fps=args.render_fps,
			seconds_per_segment=args.render_seconds_per_segment,
			output_path=final_output,
			tmp_dir=tmp_dir,
			keep_segments=args.render_keep_segments,
			extra_args=(args.render_extra_args or [])
		)
		logger.info(f"Combined video saved to: {final_output}")

	logger.info("Script finished successfully.")


if __name__ == "__main__":
	main()
