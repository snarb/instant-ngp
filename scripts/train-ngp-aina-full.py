#!/usr/bin/env python3
import sys

# Update this path to your instant-ngp 'build' directory if needed
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
   result = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, capture_output=True, text=True)
   if result.returncode != 0:
      error_msg = f"Command failed: {' '.join(map(str, cmd))} (rc={result.returncode})\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
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

      ext = Path(orig_name).suffix
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


def build_sequence(frames, palindrome=False):
   """Builds a sequence of frames for rendering. A simple forward pass, or a palindrome."""
   seq = list(frames)
   if palindrome and len(frames) > 1:
      seq = frames + frames[-2::-1]
   return seq


def render_combined_video(instant_root: Path, camera_path: Path, sequence_frames, fps: int,
                    output_path: Path, tmp_dir: Path, keep_frames: bool, extra_args):
   """
   Renders a smooth video by rendering a single frame from each snapshot
   at the corresponding point in the camera path, then stitching them together.
   """
   instant_root = Path(instant_root)
   tmp_dir = Path(tmp_dir)
   shutil.rmtree(tmp_dir)
   os.makedirs(tmp_dir, exist_ok=True)

   rendered_frame_paths = []
   total_frames_in_sequence = len(sequence_frames)

   logger.info(f"Preparing to render {total_frames_in_sequence} individual frames to create the final video.")

   for i, frame in enumerate(sequence_frames):
      # if i == 0:
      #  continue
      render_start_time = time.time()
      snapshot = frame / f"snapshot_{frame.name}.msgpack"
      if not snapshot.exists():
         alt = None
         for ext in (".msgpack", ".ingp"):
            t = frame / f"snapshot_{frame.name}{ext}"
            if t.exists():
               alt = t
               break
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

      # A temporary output pattern for the image sequence from run.py
      #temp_output_pattern = tmp_dir / f"temp_render_{i}_%04d.png"
      temp_output_pattern = tmp_dir / f"render_%04d.png"
      VIDEO_DURATION = 30 # frames_number = 40 * 30 = 1200
      total_rendered_frames_cnt = fps * VIDEO_DURATION
      frames_per_segment = int(total_rendered_frames_cnt / len(sequence_frames))
      START_FRAME = frames_per_segment * i
      END_FRAME = frames_per_segment * (i + 1)
      #SEGMENG_DURATION = 1
      # FRAMES_PER_SEGMENT = 2
      # START_FRAME = FRAMES_PER_SEGMENT * i
      # END_FRAME = FRAMES_PER_SEGMENT * (i + 1)
      #video_duration = i * 1 # 1 sec per frame
      video_duration = len(sequence_frames) * 1
      frames_per_segment = fps * 1
      START_FRAME = frames_per_segment * i
      END_FRAME = frames_per_segment * (i + 1)
      render_cmd = [
         sys.executable,
         str(instant_root / "scripts/run-ngp-aina.py"),
         "--scene", str(frame),
         "--load_snapshot", str(snapshot),
         "--video_camera_path", str(camera_path),
         "--video_fps", str(fps),
         # Set n_seconds so that t = frame_index / total_frames is correctly computed inside run.py
         "--video_n_seconds", str(video_duration),
         # KEY CHANGE: Render ONLY the single frame at index `i` from the camera path
         "--video_render_range", str(START_FRAME), str(END_FRAME),
         # KEY CHANGE: Output to an image sequence format. run.py will fill in the frame number.
         "--video_output", str(temp_output_pattern),
		 "--aabb", "0.39", "0.13", "0.25", "0.65", "0.70", "0.67"

      ]
      if extra_args:
         render_cmd += list(extra_args)

      logger.info(f"Rendering frame {i + 1}/{total_frames_in_sequence} using model from {frame.name}")
      run(render_cmd, cwd=str(instant_root))

      # The expected output filename will have the frame index 'i' filled in by run.py
      #expected_source_frame = Path(str(temp_output_pattern) % i)
      #final_frame_path = tmp_dir / f"final_{i:04d}.png"

      # if not expected_source_frame.exists():
      #    logger.warning(f"Rendered frame not produced for model {frame.name}: expected {expected_source_frame}")
      #    continue

      # Move the rendered frame to its final, consistently named location for ffmpeg
      #shutil.move(str(expected_source_frame), str(final_frame_path))
      final_frame_path = temp_output_pattern
      rendered_frame_paths.append(final_frame_path)

      render_duration = time.time() - render_start_time
      logger.info(f"Frame {i+1} rendered in {render_duration:.2f} seconds.")

   if not rendered_frame_paths:
      logger.error("No frames were rendered; cannot create video.")
      return

   logger.info(f"Stitching {len(rendered_frame_paths)} frames into final video: {output_path}")
   # KEY CHANGE: Use ffmpeg to create a video from the sequence of rendered PNG images
   ffmpeg_cmd = [
      "ffmpeg", "-y",
      "-framerate", str(fps),
      "-i", str(tmp_dir / "render_%04d.png"),
      "-c:v", "libx264",
      "-pix_fmt", "yuv420p",
      str(output_path)
   ]
   run(ffmpeg_cmd, cwd=None)

   if not keep_frames:
      logger.info("Cleaning up temporary rendered frames.")
      for p in rendered_frame_paths:
         try:
            p.unlink()
         except Exception:
            pass
      try:
         # Try to remove the directory if it's empty
         tmp_dir.rmdir()
      except OSError:
         logger.warning(f"Could not remove temporary directory {tmp_dir} as it may not be empty.")
         pass


def main():
   parser = argparse.ArgumentParser(description="Train Instant-NGP model per frame and optionally render combined video.", add_help=True)

   # --- Core Arguments ---
   parser.add_argument("dataset_root", type=Path, help="Root directory with frame subdirectories")
   parser.add_argument("--instant_root", type=Path, default=Path.cwd(), help="Path to instant-ngp repo root (default: current dir)")

   # --- Training Arguments ---
   parser.add_argument("--first_step_n_steps", type=int, default=20000, help="Training steps for first frame (default: 20000)")
   parser.add_argument("--following_n_steps", type=int, default=10000, help="Training steps for subsequent frames (default: 10000)")
   parser.add_argument("--start_frame", type=int, default=None, help="The first frame number to start training from (inclusive).")
   parser.add_argument("--end_frame", type=int, default=None, help="The last frame number to train (inclusive).")

   # --- COLMAP Arguments ---
   parser.add_argument("--aabb_scale", type=int, help="aabb_scale arg for colmap2nerf.py")
   parser.add_argument("--mask_categories", nargs="*", help="mask_categories arg for colmap2nerf.py")
   parser.add_argument("--colmap_out", type=Path, help="Where to write transforms.json (default: first frame folder)")
   parser.add_argument("--skip_colmap", action="store_true", help="Do not run colmap (if transforms.json is already present)")

   # --- Rendering Arguments ---
   parser.add_argument("--render_only", action="store_true", help="Skip all training and colmap steps and proceed directly to rendering.")
   parser.add_argument("--render_after_train", action="store_true",
                  help="After training, render a combined video from per-frame snapshots")
   parser.add_argument("--render_camera_path", type=Path, help="Camera path JSON for video rendering (required for rendering)")
   parser.add_argument("--render_start_frame", type=int, default=None, help="First frame number to include in the render (inclusive).")
   parser.add_argument("--render_end_frame", type=int, default=None, help="Last frame number to include in the render (inclusive).")
   parser.add_argument("--render_fps", type=int, default=30, help="Output FPS for the final video (default: 30)")
   parser.add_argument("--render_palindrome", action="store_true", help="Make sequence palindrome (e.g., a-b-c-b-a), doubling the frames.")
   parser.add_argument("--render_output", type=Path, default=None, help="Final video output path (default: dataset_root/combined.mp4)")
   parser.add_argument("--render_tmpdir", type=Path, default=None,
                  help="Temporary directory for rendered frames (default: dataset_root/tmp_render)")
   parser.add_argument("--render_keep_frames", action="store_true", help="Do not delete individual rendered frames after video creation")
   parser.add_argument("--render_extra_args", nargs="*", help="Extra args for run.py during rendering (e.g. --network mynet.json)")
   parser.add_argument(
	   "--multi_camera",
	   action="store_true",
	   help="Forward to colmap2nerf.py to allow multiple cameras (different resolutions). "
			"By default colmap2nerf.py uses SINGLE camera unless this flag is present."
   )
   args, extra_args = parser.parse_known_args()
   dataset_root = args.dataset_root.resolve()

   # --- Setup Logging ---
   log_file_path = dataset_root / "training_log.txt"
   logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(message)s",
      handlers=[logging.FileHandler(log_file_path, mode='a'), logging.StreamHandler(sys.stdout)]
   )
   logger.info(f"--- Starting new script run ---")

   instant_root = args.instant_root.resolve()
   if not dataset_root.is_dir():
      logger.error(f"Dataset root does not exist or is not a directory: {dataset_root}")
      sys.exit(1)

   all_frames = try_int_sort([d for d in dataset_root.iterdir() if d.is_dir()])
   if not all_frames:
      logger.error("No frame subdirectories found in dataset_root")
      sys.exit(1)

   if not args.render_only:
      # --- Filter frames for TRAINING based on args ---
      training_frames = all_frames
      if args.start_frame is not None or args.end_frame is not None:
         logger.info(f"Filtering training frames for range: [{args.start_frame}, {args.end_frame}]")
         filtered = []
         for frame in all_frames:
            try:
               frame_num = int(frame.name)
               if args.start_frame is not None and frame_num < args.start_frame: continue
               if args.end_frame is not None and frame_num > args.end_frame: continue
               filtered.append(frame)
            except ValueError:
               logger.warning(f"Skipping non-integer frame name '{frame.name}' during filtering.")
         training_frames = filtered

      if not training_frames:
         logger.error("No frames matched the specified --start_frame/--end_frame filter. Nothing to train.")
         sys.exit(1)

      logger.info(f"Found {len(training_frames)} frames to train out of {len(all_frames)} total.")

      # --- COLMAP Stage (uses the first frame of the whole dataset as reference) ---
      reference_frame = all_frames[0]
      out_path = (args.colmap_out.resolve() if args.colmap_out else (reference_frame / "transforms.json"))
      os.makedirs(out_path.parent, exist_ok=True)
      text_folder = reference_frame / "sparse"
      os.makedirs(text_folder, exist_ok=True)

      colmap_cmd = [
         sys.executable, str(instant_root / "scripts/colmap2nerf.py"),
         "--colmap_db", str(reference_frame / "database.db"),
         "--text", str(text_folder),
         "--images", str(reference_frame),
         "--out", str(out_path),
         "--run_colmap", "--overwrite"
      ]
      if args.aabb_scale: colmap_cmd += ["--aabb_scale", str(args.aabb_scale)]
      if args.multi_camera:    colmap_cmd += ["--multi_camera"]
      if args.mask_categories: colmap_cmd += ["--mask_categories"] + list(args.mask_categories)


      if not args.skip_colmap:
         logger.info(f"Running colmap2nerf.py using reference frame: {reference_frame.name}")
         run(colmap_cmd, cwd=instant_root)
      else:
         logger.info(f"Skipping colmap. Assuming transforms.json exists at: {out_path}")

      if not out_path.exists():
         logger.error(f"FATAL: transforms.json not found at {out_path} after colmap stage.")
         sys.exit(1)

      # --- Training Stage ---
      target_dataset_transform = dataset_root / "transforms.json"
      shutil.copy2(out_path, target_dataset_transform)
      logger.info(f"Copied {out_path} -> {target_dataset_transform}")

      for frame in all_frames:  # Copy to all frames for reference
         if frame / "transforms.json" != out_path:
         	shutil.copy2(out_path, frame / "transforms.json")

      cur_step = 0
      prev_snapshot = None
      for idx, frame in enumerate(training_frames):
         frame_start_time = time.time()
         logger.info(f"--- Training frame {frame.name} ({idx + 1}/{len(training_frames)}) ---")
         fix_transforms_paths(frame / "transforms.json", frame, overwrite=True)

         snapshot_path = frame / f"snapshot_{frame.name}.msgpack"
         #is_first_ever_frame = (idx == 0) # Check index in the filtered list, not against all_frames
         is_first_ever_frame = True
         steps_to_go = args.first_step_n_steps if is_first_ever_frame else args.following_n_steps

         cur_step += steps_to_go

         run_cmd = [
            sys.executable, str(instant_root / "scripts/run2.py"),
            "--scene", str(frame),
            "--save_snapshot", str(snapshot_path),
            "--n_steps", str(cur_step)
         ]
         if not is_first_ever_frame and prev_snapshot:
            run_cmd += ["--load_snapshot", str(prev_snapshot)]
         if extra_args:
            run_cmd += extra_args

         run(run_cmd, cwd=instant_root)
         prev_snapshot = snapshot_path

         frame_duration = time.time() - frame_start_time
         logger.info(f"Finished training for {frame.name}. Took {frame_duration:.2f} seconds.")

   else:
      logger.info("--- Render-only mode enabled. Skipping all COLMAP and training steps. ---")

   # --- Rendering Stage ---
   if args.render_after_train or args.render_only:
      if not args.render_camera_path:
         logger.error("Rendering requested but --render_camera_path is missing.")
         sys.exit(1)
      if not args.render_camera_path.exists():
         logger.error(f"Camera path file not found: {args.render_camera_path}")
         sys.exit(1)

      rendering_frames = all_frames
      if args.render_start_frame is not None or args.render_end_frame is not None:
         logger.info(f"Filtering rendering frames for range: [{args.render_start_frame}, {args.render_end_frame}]")
         filtered = []
         for frame in all_frames:
            try:
               frame_num = int(frame.name)
               if args.render_start_frame is not None and frame_num < args.render_start_frame: continue
               if args.render_end_frame is not None and frame_num > args.render_end_frame: continue
               filtered.append(frame)
            except ValueError:
               logger.warning(f"Skipping non-integer frame name '{frame.name}' during render filtering.")
         rendering_frames = filtered

      if not rendering_frames:
         logger.error("No frames matched the specified render frame filter. Nothing to render.")
         sys.exit(1)

      logger.info(f"Found {len(rendering_frames)} frames to render.")

      final_output = args.render_output.resolve() if args.render_output else (dataset_root / "combined.mp4")
      tmp_dir = (args.render_tmpdir.resolve() if args.render_tmpdir else (dataset_root / "tmp_render"))
      tmp_dir = Path(tmp_dir)
      if tmp_dir.exists() and tmp_dir.is_dir():
       shutil.rmtree(tmp_dir)
      tmp_dir.mkdir(parents=True, exist_ok=True)
      seq_frames = build_sequence(rendering_frames, palindrome=args.render_palindrome)
      logger.info(f"Rendering a video from {len(seq_frames)} total frames/models.")

      render_combined_video(
         instant_root=instant_root,
         camera_path=args.render_camera_path.resolve(),
         sequence_frames=seq_frames,
         fps=args.render_fps,
         output_path=final_output,
         tmp_dir=tmp_dir,
         keep_frames=args.render_keep_frames,
         extra_args=(args.render_extra_args or [])
      )
      logger.info(f"Combined video saved to: {final_output}")

   logger.info("Script finished successfully.")


if __name__ == "__main__":
   main()
