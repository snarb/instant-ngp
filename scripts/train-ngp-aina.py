#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Convenience wrapper around ``scripts/run.py`` for training pipelines.

This script only extends the default training entry point with optional
command line toggles for NVIDIA DLSS so that renders produced during
training (e.g. validation videos) can benefit from the super sampling
pass when supported by the system.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Launch training with optional DLSS support by forwarding arguments to scripts/run.py.",
        add_help=True,
    )
    parser.add_argument(
        "--dlss",
        action="store_true",
        help="Enable NVIDIA DLSS when rendering outputs. Requires GPU and driver support.",
    )
    parser.add_argument(
        "--dlss_mode",
        choices=["ultra_performance", "max_performance", "balanced", "max_quality", "ultra_quality", "dlaa"],
        default=None,
        help="Force a DLSS quality preset (including DLAA for anti-aliasing).",
    )
    parser.add_argument(
        "--dlss_sharpening",
        "--dlss-sharpening",
        type=float,
        default=None,
        help="Override the DLSS sharpening amount (0.0 - 1.0).",
    )

    args, remaining = parser.parse_known_args()
    return args, remaining


def _append_if_missing(cmd: List[str], flag: str) -> None:
    if flag not in cmd:
        cmd.append(flag)


def main() -> None:
    args, remaining = _parse_args()

    run_script = os.path.join(os.path.dirname(__file__), "run.py")

    command: List[str] = [sys.executable, run_script, "--train"]
    command.extend(remaining)

    if args.dlss:
        _append_if_missing(command, "--dlss")

    if args.dlss_mode and "--dlss_mode" not in command:
        _append_if_missing(command, "--dlss")
        command.extend(["--dlss_mode", args.dlss_mode])

    if args.dlss_sharpening is not None and "--dlss_sharpening" not in command and "--dlss-sharpening" not in command:
        command.extend(["--dlss_sharpening", str(args.dlss_sharpening)])

    subprocess.check_call(command)


if __name__ == "__main__":
    main()
