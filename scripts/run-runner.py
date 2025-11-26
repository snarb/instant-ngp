#!/usr/bin/env python3
"""
Simple fixed runner that calls scripts/run-ngp-aina.py
using constant parameters (no argparse).
"""

import subprocess
import shlex

WRAPPER = "scripts/run-ngp-aina.py"

# ------------------------------
# CONSTANT CONFIG (EDIT HERE)
# ------------------------------

SCENE = "/fsx/Test_Prism_new/Walking1/masked_jpgs/sequence1/seqA1/JPEG_FR/000000/transforms_train.json"
TEST_SCENE = "/fsx/Test_Prism_new/Walking1/masked_jpgs/sequence1/seqA1/JPEG_FR/000000/transforms_test.json"

N_STEPS = 40000
VAL_INTERVAL = 30000
VAL_MAX_IMAGES = 3

TRAIN_MODE = "nerf"

USE_WANDB = True
WANDB_PROJECT = "instant_ngp_aina"
WANDB_EXPERIMENT = "walking1_scale4_masked_high_res"
NETWORK_CONFIG = "configs/nerf/big_v2.json"

USE_FACE_CROP = True
LOG_VAL_IMAGES = True

# ------------------------------
# BUILD CMD
# ------------------------------

cmd_parts = [
    "/home/ubuntu/anaconda3/envs/ngp/bin/python3", WRAPPER,
    "--scene", SCENE,
    "--test_transforms", TEST_SCENE,
    "--n_steps", str(N_STEPS),
    "--val_interval", str(VAL_INTERVAL),
    "--val_max_images", str(VAL_MAX_IMAGES),
    "--train_mode", TRAIN_MODE,
    "--train_mode", TRAIN_MODE,
    "--width", str(1920),
    "--height", str(1080),
    "--network", str(1080),
]

if LOG_VAL_IMAGES:
    cmd_parts.append("--val_log_images")

if NETWORK_CONFIG:
    cmd_parts += [
        "--network", NETWORK_CONFIG,
    ]


if USE_WANDB:
    cmd_parts += [
        "--wandb",
        "--wandb_project", WANDB_PROJECT,
        "--wandb_experiment", WANDB_EXPERIMENT,
    ]

if USE_FACE_CROP:
    cmd_parts.append("--face_crop")

# ------------------------------
# RUN
# ------------------------------

cmd = " ".join(shlex.quote(x) for x in cmd_parts)

print("\nRunning (via runner):\n", cmd, "\n")
subprocess.run(cmd, shell=True)
