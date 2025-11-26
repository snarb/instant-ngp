import sys
pyngp_path = '/home/ubuntu/repos/instant-ngp/build/'
sys.path.append(pyngp_path)
import pyngp as ngp
import argparse
import shutil
import subprocess
import json
from pathlib import Path

def run(cmd, cwd=None):
    print("Running:", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(map(str, cmd))} (rc={result.returncode})")

def try_int_sort(paths):
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

    Args:
        json_path: Path to transforms.json to fix.
        source_dir: Directory whose name is the frame number (Path).
        overwrite: If True, overwrite the json_path file. If False, returns modified dict.
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
        # extract filename from whatever path is present
        orig_name = Path(orig_fp).name

        # use suffix() and stem to reliably split extension and last-dot part
        ext = Path(orig_name).suffix  # includes leading dot, e.g. '.jpg'
        stem = orig_name[:-len(ext)] if ext else orig_name  # e.g. 'A003_A016_0501OD.002361'

        if '.' in stem:
            prefix, old_frame = stem.rsplit('.', 1)
        else:
            # fallback: no dot found in stem, treat whole stem as prefix
            prefix, old_frame = stem, None

        # build new filename with the frame_name as the suffix
        if prefix:
            new_name = f"{prefix}.{frame_name}{ext}"
        else:
            # extreme fallback (shouldn't happen for your format)
            new_name = f"{frame_name}{ext}"

        # final absolute path inside source_dir
        new_fp = str((source_dir / new_name).resolve())

        if frame.get("file_path") != new_fp:
            frame["file_path"] = new_fp
            changed = True

    if overwrite and changed:
        with json_path.open("w") as f:
            json.dump(data, f, indent=2)

    return data

def main():
    parser = argparse.ArgumentParser(description="Train Instant-NGP model per frame.")
    parser.add_argument("dataset_root", type=Path, help="Root directory with frame subdirectories")
    parser.add_argument("--instant_root", type=Path, default=Path.cwd(), help="Path to instant-ngp repo root (default: current dir)")
    parser.add_argument("--first_step_n_steps", type=int, default=40000, help="Training steps for first frame (default: 20000)")
    parser.add_argument("--following_n_steps", type=int, default=5000, help="Training steps for subsequent frames (default: 2000)")
    parser.add_argument("--aabb_scale", type=int, help="aabb_scale arg for colmap2nerf.py")
    parser.add_argument("--mask_categories", nargs="*", help="mask_categories arg for colmap2nerf.py")
    parser.add_argument("--colmap_out", type=Path, help="Where to write transforms.json (default: first frame folder)")
    parser.add_argument("--run_args", nargs=argparse.REMAINDER, help="Extra args forwarded to run.py")
    parser.add_argument("--skip_colmap", action="store_true", help="do not run the colmap (if transforms.json is already present)")

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    instant_root = args.instant_root.resolve()
    frames = [d for d in dataset_root.iterdir() if d.is_dir()]
    frames = try_int_sort(frames)
    if not frames:
        sys.exit("No frame subdirectories found")

    # Default --out is the first frame folder's transforms.json (as requested)
    out_path = (args.colmap_out.resolve() if args.colmap_out else (frames[0] / "transforms.json"))

    # 1) Run colmap2nerf.py for the first frame only
    colmap_cmd = [
        sys.executable,
        str(instant_root / "scripts/colmap2nerf.py"),
        "--colmap_db", str(frames[0] / "database.db"),
        "--text", str(frames[0] / "sparse/0"),
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
        run(colmap_cmd, cwd=instant_root)

    if not out_path.exists():
        sys.exit(f"Expected transforms.json not found at {out_path}")

    # Copy transforms.json (replace) to dataset root and into each frame subdir
    target_dataset_transform = dataset_root / "transforms.json"
    shutil.copy2(out_path, target_dataset_transform)

    for frame in frames[1:]:
        shutil.copy2(out_path, frame / "transforms.json")

    # 2) Train per frame
    cur_step = 0
    prev_snapshot = None
    for idx, frame in enumerate(frames):
        print(f"Processing {frame} {idx} from {len(frames)}")
        # ensure each frame's transforms.json points to images inside that frame dir
        frame_transform = frame / "transforms.json"
        try:
            fix_transforms_paths(frame_transform, frame, overwrite=True)
        except Exception as e:
            print(f"Warning: failed to fix transforms for {frame_transform}: {e}")

        snapshot_path = frame / f"snapshot_{frame.name}.msgpack"
        steps_to_go = args.first_step_n_steps if idx == 0 else args.following_n_steps
        cur_step += steps_to_go
        run_cmd = [ # cur_step += steps_to_go
            sys.executable,
            str(instant_root / "scripts/run.py"),
            "--scene", str(frame),
            "--save_snapshot", str(snapshot_path),
            "--n_steps", str(cur_step)
        ]
        if idx > 0 and prev_snapshot:
            run_cmd += ["--load_snapshot", str(prev_snapshot)]
        if args.run_args:
            run_cmd += args.run_args

        run(run_cmd, cwd=instant_root)
        prev_snapshot = snapshot_path
        print(f"Done {frame}")

if __name__ == "__main__":
    main()
