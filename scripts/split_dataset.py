#!/usr/bin/env python3
"""
Split instant-NGP transforms.json into train/test sets.

Supports:
  - Single-folder mode: splits only the transforms.json in --root.
  - Recursive mode (--recursive): splits all subfolders containing transforms.json.

Features:
  - Always uses the same random seed across all folders (unless --indices is used).
  - Preserves all top-level fields except 'frames'.
  - Deterministic and reproducible.
  - Can run in dry-run mode to preview actions.
  - Supports explicit test indices via --indices.

Example:
  python split_transforms.py \
    --root /fsx/.../JPEG_FR \
    --val-size 50 \
    --seed 42 \
    --recursive

  python split_transforms.py \
    --root /fsx/.../JPEG_FR \
    --indices 0,5,6 \
    --recursive
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional


# ---------- Utility I/O ----------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- Core split logic ----------

def split_transforms(
    transforms: Dict[str, Any],
    val_size: int,
    rng: random.Random,
    explicit_indices: Optional[List[int]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:

    # Sort deterministically
    frames: List[Dict[str, Any]] = sorted(
        transforms.get("frames", []),
        key=lambda f: f.get("file_path", "")
    )
    n = len(frames)
    if n == 0:
        raise ValueError("Input transforms has no frames.")

    # ---------- NEW FEATURE: explicit indices ----------
    if explicit_indices is not None:
        idx_set = set(explicit_indices)
        if any(i < 0 or i >= n for i in idx_set):
            raise ValueError(
                f"Some indices out of range 0..{n-1}: {explicit_indices}"
            )
        test_indices = idx_set
    else:
        # random selection mode
        if not 0 <= val_size <= n:
            raise ValueError(f"--val-size ({val_size}) must be between 0 and {n}.")
        test_indices = set(rng.sample(range(n), k=val_size))

    # Split
    train_frames = [f for i, f in enumerate(frames) if i not in test_indices]
    test_frames = [f for i, f in enumerate(frames) if i in test_indices]

    base = {k: v for k, v in transforms.items() if k != "frames"}

    train_json = dict(base, frames=train_frames)
    test_json = dict(base, frames=test_frames)

    return train_json, test_json


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split instant-NGP transforms.json into train/test sets.")
    p.add_argument("--root", required=True, type=Path, help="Path to dataset root or single folder")
    p.add_argument("--val-size", type=int, default=0, help="Number of frames for test split")
    p.add_argument("--indices", type=str, default=None,
                   help="Comma-separated list of explicit test-set indices (overrides random split)")
    p.add_argument("--filename", type=str, default="transforms.json", help="Transforms filename")
    p.add_argument("--recursive", action="store_true", help="Recursively process subfolders containing transforms.json")
    p.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    return p.parse_args()


def parse_indices(arg: Optional[str]) -> Optional[List[int]]:
    if arg is None:
        return None
    if arg.strip() == "":
        return None
    return [int(x) for x in arg.split(",")]


def process_folder(folder: Path, args: argparse.Namespace, explicit_indices: Optional[List[int]]) -> bool:
    rng = random.Random(42)

    file = folder / args.filename
    if not file.exists():
        return False

    try:
        transforms = load_json(file)
        train_json, test_json = split_transforms(
            transforms, args.val_size, rng, explicit_indices=explicit_indices
        )
    except Exception as e:
        print(f"[ERROR] {folder}: {e}")
        return False

    out_train = folder / "transforms_train.json"
    out_test = folder / "transforms_test.json"

    if args.dry_run:
        print(f"[DRY] {folder} → train={len(train_json['frames'])}, test={len(test_json['frames'])}")
    else:
        write_json(out_train, train_json)
        write_json(out_test, test_json)
        print(f"[OK] {folder} → train={len(train_json['frames'])}, test={len(test_json['frames'])}")
    return True


def main() -> None:
    args = parse_args()
    explicit_indices = parse_indices(args.indices)

    if explicit_indices is not None:
        print(f"Using explicit indices: {explicit_indices}")
        print("⚠️  --val-size is ignored.")

    if not args.root.exists():
        raise SystemExit(f"Root not found: {args.root}")

    if args.recursive:
        folders = sorted({p.parent for p in args.root.rglob(args.filename)})
    else:
        folders = [args.root]

    if not folders:
        print(f"No '{args.filename}' found under {args.root}")
        return

    print(f"Processing {len(folders)} folder(s) under {args.root}")

    count = 0
    for folder in folders:
        if process_folder(folder, args, explicit_indices):
            count += 1

    print(f"Done. Successfully processed {count}/{len(folders)} folders.")


if __name__ == "__main__":
    main()
