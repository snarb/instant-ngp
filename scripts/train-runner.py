"""
Convenient runner wrapper for train-ngp-aina.py.
Allows easy modification of arguments in one place.
"""

import subprocess
import shlex

# ------------------------------
# CONFIG SECTION — EDIT FREELY
# ------------------------------

# SCENE = "/fsx/Test_Prism_new/Walking1/base_jpgs/sequence1/seqA1/JPEG_FR/000000/transforms_train.json"
# TEST_SCENE = "/fsx/Test_Prism_new/Walking1/base_jpgs/sequence1/seqA1/JPEG_FR/000000/transforms_test.json"

SCENE = "/fsx/Test_Prism_new/Walking1/base_jpgs/sequence1/seqA1/JPEG_FR/000000/transforms_train.json"
TEST_SCENE = "/fsx/Test_Prism_new/Walking1/base_jpgs/sequence1/seqA1/JPEG_FR/000000/transforms_test.json"

N_STEPS = 40000
VAL_INTERVAL = 10000
VAL_MAX_IMAGES = 3

TRAIN_MODE = "nerf"

USE_WANDB = True
WANDB_PROJECT = "instant_ngp_aina"
WANDB_EXPERIMENT = "walking1_cropped"
WANDB_EXPERIMENT = "walking1_scale4"

USE_FACE_CROP = True
LOG_VAL_IMAGES = True

# ------------------------------
# BUILD ARGUMENT LIST
# ------------------------------

cmd_parts = [
    "/home/ubuntu/anaconda3/envs/ngp/bin/python3", "train-ngp-aina.py",
    "--scene", SCENE,
    "--test_transforms", TEST_SCENE,
    "--n_steps", str(N_STEPS),
    "--val_interval", str(VAL_INTERVAL),
    "--val_max_images", str(VAL_MAX_IMAGES),
    "--train_mode", TRAIN_MODE,
]

if LOG_VAL_IMAGES:
    cmd_parts.append("--val_log_images")

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
print("\nRunning command:\n", cmd, "\n")

subprocess.run(cmd, shell=True)
