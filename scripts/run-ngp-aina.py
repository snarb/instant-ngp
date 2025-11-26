#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp  # noqa

# --- NEW / W&B + face-crop helpers ---

try:
    import wandb  # noqa

    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

try:
    import cv2  # noqa

    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def init_wandb_from_args(args):
    """Initialize Weights & Biases run if enabled and available."""
    if not getattr(args, "wandb", False):
        return None

    if not HAS_WANDB:
        print("[W&B] wandb package not installed, disabling W&B logging.")
        return None

    run = wandb.init(
        project=args.wandb_project or "instant-ngp",
        name=args.wandb_experiment or None,
        config={
            "scene": args.scene,
            "network": args.network,
            "n_steps": args.n_steps,
            "nerf_compatibility": args.nerf_compatibility,
            "train_mode": args.train_mode,
            "rfl_warmup_steps": args.rfl_warmup_steps,
            "rflrelax_begin_step": args.rflrelax_begin_step,
            "rflrelax_end_step": args.rflrelax_end_step,
            "val_interval": args.val_interval,
            "val_max_images": args.val_max_images,
            "face_crop": args.face_crop,
            "n_max": args.n_max,
            "hash_table_size": args.hash_table_size,
        },
    )
    print("[W&B] Logging enabled.")
    return run


def init_face_cascade_from_args(args):
    """Initialize OpenCV face cascade if face_crop is enabled."""
    if not getattr(args, "face_crop", False):
        return None

    if not HAS_CV2:
        print("[face-crop] OpenCV (cv2) not installed, disabling face-crop.")
        return None

    cascade_path = args.face_cascade or os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml",
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(
            f"[face-crop] Failed to load face cascade from {cascade_path}, disabling face-crop."
        )
        return None

    print(f"[face-crop] Using cascade {cascade_path}")
    return face_cascade


def detect_primary_face_bbox(face_cascade, img_rgb_uint8):
    """Return (x, y, w, h) of the largest detected face or None."""
    if face_cascade is None:
        return None
    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return None
    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def evaluate_current_dataset(
    testbed,
    split_name,
    max_images=0,
    log_first_images=True,
    wb_run=None,
    step=None,
    enable_face_crop=False,
    face_cascade=None,
    save_prefix=None,
):
    """
    Compute PSNR/SSIM (and optional face-crop metrics) on the dataset
    currently loaded into testbed.nerf.training.dataset.
    """
    n_images = testbed.nerf.training.dataset.n_images
    if n_images == 0:
        print(f"[VAL/{split_name}] Dataset has 0 images, skipping.")
        return {}

    if max_images > 0:
        n_eval = min(max_images, n_images)
    else:
        n_eval = n_images

    print(f"[VAL/{split_name}] Evaluating {n_eval}/{n_images} images.")

    totmse = 0.0
    totpsnr = 0.0
    totssim = 0.0
    totcount = 0
    minpsnr = 1e9
    maxpsnr = 0.0

    face_totmse = 0.0
    face_totpsnr = 0.0
    face_totssim = 0.0
    face_count = 0

    first_ref_u8 = None
    first_out_u8 = None
    first_face_ref_u8 = None
    first_face_out_u8 = None

    spp = 8
    testbed.nerf.render_min_transmittance = 1e-4
    testbed.snap_to_pixel_centers = True
    testbed.render_with_lens_distortion = True

    for i in range(n_eval):
        resolution = testbed.nerf.training.dataset.metadata[i].resolution
        testbed.render_ground_truth = True
        testbed.set_camera_to_training_view(i)
        ref_image = testbed.render(resolution[0], resolution[1], 1, True)

        testbed.render_ground_truth = False
        image = testbed.render(resolution[0], resolution[1], spp, True)

        if log_first_images and i == 0:
            prefix = save_prefix or split_name
            write_image(f"{prefix}_ref.png", ref_image)
            write_image(f"{prefix}_out.png", image)
            diffimg = np.absolute(image - ref_image)
            diffimg[..., 3:4] = 1.0
            write_image(f"{prefix}_diff.png", diffimg)

        A = np.clip(linear_to_srgb(image[..., :3]), 0.0, 1.0)
        R = np.clip(linear_to_srgb(ref_image[..., :3]), 0.0, 1.0) if False else np.clip(  # safety; we want linear_to_srgb
            linear_to_srgb(ref_image[..., :3]), 0.0, 1.0
        )

        mse = float(compute_error("MSE", A, R))
        ssim = float(compute_error("SSIM", A, R))
        psnr = mse2psnr(mse)

        totmse += mse
        totssim += ssim
        totpsnr += psnr
        minpsnr = psnr if psnr < minpsnr else minpsnr
        maxpsnr = psnr if psnr > maxpsnr else maxpsnr
        totcount += 1

        if log_first_images and i == 0:
            first_ref_u8 = (R * 255.0).astype(np.uint8)
            first_out_u8 = (A * 255.0).astype(np.uint8)

        # Face-crop metrics
        if enable_face_crop and face_cascade is not None and HAS_CV2:
            ref_u8 = (R * 255.0).astype(np.uint8)
            bbox = detect_primary_face_bbox(face_cascade, ref_u8)
            if bbox is not None:
                x, y, w, h = bbox
                x2 = min(x + w, ref_u8.shape[1])
                y2 = min(y + h, ref_u8.shape[0])
                x = max(x, 0)
                y = max(y, 0)
                if x2 > x and y2 > y:
                    A_face = A[y:y2, x:x2, :]
                    R_face = R[y:y2, x:x2, :]
                    if A_face.size and R_face.size:
                        mse_f = float(compute_error("MSE", A_face, R_face))
                        ssim_f = float(compute_error("SSIM", A_face, R_face))
                        psnr_f = mse2psnr(mse_f)
                        face_totmse += mse_f
                        face_totssim += ssim_f
                        face_totpsnr += psnr_f
                        face_count += 1

                        if log_first_images and first_face_ref_u8 is None:
                            first_face_ref_u8 = (R_face * 255.0).astype(np.uint8)
                            first_face_out_u8 = (A_face * 255.0).astype(np.uint8)

    if totcount == 0:
        print(f"[VAL/{split_name}] No images evaluated.")
        return {}

    psnr_avgmse = mse2psnr(totmse / totcount)
    psnr_mean = totpsnr / totcount
    ssim_mean = totssim / totcount

    print(
        f"[VAL/{split_name}] PSNR(mean)={psnr_mean:.3f} "
        f"[min={minpsnr:.3f} max={maxpsnr:.3f}] "
        f"PSNR(avg_mse)={psnr_avgmse:.3f} SSIM={ssim_mean:.4f}"
    )

    metrics = {
        f"{split_name}/psnr": float(psnr_mean),
        f"{split_name}/psnr_avgmse": float(psnr_avgmse),
        f"{split_name}/psnr_min": float(minpsnr),
        f"{split_name}/psnr_max": float(maxpsnr),
        f"{split_name}/ssim": float(ssim_mean),
        f"{split_name}/count": int(totcount),
    }

    if face_count > 0:
        face_psnr_avgmse = mse2psnr(face_totmse / face_count)
        face_psnr_mean = face_totpsnr / face_count
        face_ssim_mean = face_totssim / face_count
        print(
            f"[VAL/{split_name}/face] PSNR(mean)={face_psnr_mean:.3f} "
            f"PSNR(avg_mse)={face_psnr_avgmse:.3f} SSIM={face_ssim_mean:.4f} "
            f"(n={face_count})"
        )
        metrics.update(
            {
                f"{split_name}/face_psnr": float(face_psnr_mean),
                f"{split_name}/face_psnr_avgmse": float(face_psnr_avgmse),
                f"{split_name}/face_ssim": float(face_ssim_mean),
                f"{split_name}/face_count": int(face_count),
            }
        )

    if wb_run is not None and HAS_WANDB:
        log_payload = dict(metrics)
        if log_first_images and first_ref_u8 is not None and first_out_u8 is not None:
            log_payload[f"{split_name}/ref_image"] = wandb.Image(first_ref_u8)
            log_payload[f"{split_name}/out_image"] = wandb.Image(first_out_u8)
        if (
            enable_face_crop
            and first_face_ref_u8 is not None
            and first_face_out_u8 is not None
        ):
            log_payload[f"{split_name}/face_ref_image"] = wandb.Image(first_face_ref_u8)
            log_payload[f"{split_name}/face_out_image"] = wandb.Image(first_face_out_u8)

        wandb_step = int(step) if step is not None else int(testbed.training_step)
        wb_run.log(log_payload, step=wandb_step)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run instant neural graphics primitives with additional configuration & output options"
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.",
    )

    parser.add_argument(
        "--scene",
        "--training_data",
        default="",
        help=(
            "The scene to load. Can be the scene's name or a full path to the training data. "
            "Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume."
        ),
    )
    parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS)  # deprecated
    parser.add_argument(
        "--network", default="", help="Path to the network config. Uses the scene's default if unspecified."
    )

    parser.add_argument(
        "--load_snapshot",
        "--snapshot",
        default="",
        help="Load this snapshot before training. recommended extension: .ingp/.msgpack",
    )
    parser.add_argument(
        "--save_snapshot",
        default="",
        help="Save this snapshot after training. recommended extension: .ingp/.msgpack",
    )

    parser.add_argument(
        "--nerf_compatibility",
        action="store_true",
        help=(
            "Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, "
            "but helps with high PSNR on synthetic scenes."
        ),
    )
    parser.add_argument(
        "--test_transforms",
        default="",
        help="Path to a nerf style transforms json from which we will compute PSNR.",
    )
    parser.add_argument(
        "--near_distance",
        default=-1,
        type=float,
        help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default",
    )
    parser.add_argument(
        "--exposure",
        default=0.0,
        type=float,
        help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.",
    )

    parser.add_argument(
        "--train_mode",
        default="",
        type=str,
        help="The training mode to use. Can be 'nerf', 'rfl', 'rfl_relax'. If not specified, the default mode will be used.",
    )
    parser.add_argument(
        "--rfl_warmup_steps",
        type=int,
        default=1000,
        help=(
            "Number of steps to train in NeRF mode before switching to RFL mode. "
            "Default is 1000. Only used if --train_mode is set to 'rfl'."
        ),
    )
    parser.add_argument(
        "--rflrelax_begin_step",
        type=int,
        default=15000,
        help=(
            "First training step in which RflRelax mode is used. "
            "Default is 15000. Only used if --train_mode is set to 'rflrelax'."
        ),
    )
    parser.add_argument(
        "--rflrelax_end_step",
        type=int,
        default=30000,
        help=(
            "Last training step in which RflRelax mode is used. "
            "Default is 30000. Only used if --train_mode is set to 'rflrelax'."
        ),
    )

    parser.add_argument(
        "--screenshot_transforms",
        default="",
        help="Path to a nerf style transforms.json from which to save screenshots.",
    )
    parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
    parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
    parser.add_argument(
        "--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots."
    )

    parser.add_argument(
        "--video_camera_path",
        default="",
        help="The camera path to render, e.g., base_cam.json.",
    )
    parser.add_argument(
        "--video_camera_smoothing",
        action="store_true",
        help=(
            "Applies additional smoothing to the camera trajectory with the caveat that the endpoint "
            "of the camera path may not be reached."
        ),
    )
    parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
    parser.add_argument(
        "--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long."
    )
    parser.add_argument(
        "--video_render_range",
        type=int,
        nargs=2,
        default=(-1, -1),
        metavar=("START_FRAME", "END_FRAME"),
        help="Limit output to frames between START_FRAME and END_FRAME (inclusive)",
    )
    parser.add_argument(
        "--video_spp",
        type=int,
        default=8,
        help="Number of samples per pixel. A larger number means less noise, but slower rendering.",
    )
    parser.add_argument(
        "--video_output",
        type=str,
        default="video.mp4",
        help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).",
    )

    parser.add_argument(
        "--save_mesh",
        default="",
        help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.",
    )
    parser.add_argument(
        "--marching_cubes_res",
        default=256,
        type=int,
        help="Sets the resolution for the marching cubes grid.",
    )
    parser.add_argument(
        "--marching_cubes_density_thresh",
        default=2.5,
        type=float,
        help="Sets the density threshold for marching cubes.",
    )

    parser.add_argument(
        "--width",
        "--screenshot_w",
        type=int,
        default=0,
        help="Resolution width of GUI and screenshots.",
    )
    parser.add_argument(
        "--height",
        "--screenshot_h",
        type=int,
        default=0,
        help="Resolution height of GUI and screenshots.",
    )

    parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="If the GUI is enabled, controls whether training starts immediately.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=-1,
        help="Number of steps to train for before quitting.",
    )
    parser.add_argument(
        "--second_window",
        action="store_true",
        help="Open a second window containing a copy of the main output.",
    )
    parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")

    parser.add_argument(
        "--sharpen",
        default=0,
        help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.",
    )
    parser.add_argument(
        "--aabb",
        nargs=6,
        type=float,
        metavar=("min_x", "min_y", "min_z", "max_x", "max_y", "max_z"),
        help="Axis-aligned bounding box (min_x, min_y, min_z, max_x, max_y, max_z)",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default="",
        help="Path to ffmpeg executable. If not specified, uses system-wide ffmpeg.",
    )
    parser.add_argument(
        "--override_training_step",
        type=int,
        default=None,
        help="Set the training step after loading a snapshot. Useful for resuming with a different scheduler state.",
    )

    # --- NEW / Validation & W&B & face-crop ---
    parser.add_argument(
        "--val_interval",
        type=int,
        default=0,
        help="Run validation (train+test) every N training steps. 0 disables periodic validation.",
    )
    parser.add_argument(
        "--val_max_images",
        type=int,
        default=0,
        help="Max number of images per split for validation (0 = use all).",
    )
    parser.add_argument(
        "--val_log_images",
        action="store_true",
        help="Log reference/output images during validation (first image per split).",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_experiment",
        type=str,
        default="",
        help="W&B experiment / run name.",
    )

    parser.add_argument(
        "--face_crop",
        action="store_true",
        help="Enable face-only metrics using OpenCV face detector.",
    )
    parser.add_argument(
        "--face_cascade",
        type=str,
        default="",
        help="Optional path to Haar cascade XML; defaults to cv2.data.haarcascades frontalface.",
    )

    # --- NEW / N_max and hash table size arguments ---
    parser.add_argument(
        "--n_max",
        type=int,
        default=-1,
        help=(
            "Maximum number of samples per ray during training. "
            "Higher values allow more detailed scenes but use more memory. "
            "Typical values: 512-2048. Use -1 to keep the default."
        ),
    )
    parser.add_argument(
        "--hash_table_size",
        type=int,
        default=-1,
        help=(
            "Size of the hash table (log2). Controls the capacity of the feature grid. "
            "Typical values: 19 (2^19 = ~524k entries) to 21 (2^21 = ~2M entries). "
            "Use -1 to keep the default (typically 19)."
        ),
    )

    return parser.parse_args()


def get_scene(scene):
    for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
        if scene in scenes:
            return scenes[scene]
    return None


if __name__ == "__main__":
    args = parse_args()
    if args.vr:  # VR implies having the GUI running at the moment
        args.gui = True

    wb_run = init_wandb_from_args(args)
    face_cascade = init_face_cascade_from_args(args)

    if args.mode:
        print(
            "Warning: the '--mode' argument is no longer in use. It has no effect. "
            "The mode is automatically chosen based on the scene."
        )

    # Determine ffmpeg path
    ffmpeg_path = args.ffmpeg_path if args.ffmpeg_path else "ffmpeg"

    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR

    for file in args.files:
        scene_info = get_scene(file)
        if scene_info:
            file = os.path.join(scene_info["data_dir"], scene_info["dataset"])
        testbed.load_file(file)

    if args.scene:
        scene_info = get_scene(args.scene)
        if scene_info is not None:
            args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
            if not args.network and "network" in scene_info:
                args.network = scene_info["network"]

        testbed.load_training_data(args.scene)

    if args.gui:
        # Pick a sensible GUI resolution depending on arguments.
        sw = args.width or 1920
        sh = args.height or 1080
        while sw * sh > 1920 * 1080 * 4:
            sw = int(sw / 2)
            sh = int(sh / 2)
        testbed.init_window(sw, sh, second_window=args.second_window)
        if args.vr:
            testbed.init_vr()

    if args.load_snapshot:
        scene_info = get_scene(args.load_snapshot)
        if scene_info is not None:
            args.load_snapshot = default_snapshot_filename(scene_info)
        testbed.load_snapshot(args.load_snapshot)
    elif args.network:
        testbed.reload_network_from_file(args.network)

    if args.override_training_step is not None:
        if args.override_training_step < 0:
            raise ValueError("--override_training_step must be non-negative")
        testbed.training_step = args.override_training_step

    ref_transforms = {}
    if args.screenshot_transforms:  # try to load the given file straight away
        print("Screenshot transforms from ", args.screenshot_transforms)
        with open(args.screenshot_transforms) as f:
            ref_transforms = json.load(f)

    if testbed.mode == ngp.TestbedMode.Sdf:
        testbed.tonemap_curve = ngp.TonemapCurve.ACES

    testbed.nerf.sharpen = float(args.sharpen)
    testbed.exposure = args.exposure
    testbed.shall_train = args.train if args.gui else True

    network_stem = (
        os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"
    )
    if testbed.mode == ngp.TestbedMode.Sdf:
        setup_colored_sdf(testbed, args.scene)

    if args.near_distance >= 0.0:
        print("NeRF training ray near_distance ", args.near_distance)
        testbed.nerf.training.near_distance = args.near_distance

    if args.n_max > 0:
        print(f"Setting N_max (max samples per ray) to {args.n_max}")
        testbed.nerf.training.n_max = args.n_max

    if args.hash_table_size > 0:
        print(f"Setting hash table size to 2^{args.hash_table_size} = {2**args.hash_table_size} entries")
        testbed.nerf.training.hash_table_size = args.hash_table_size

    if args.train_mode:
        if args.train_mode.lower() == "nerf":
            testbed.nerf.training.train_mode = ngp.TrainMode.Nerf
        elif args.train_mode.lower() == "rfl":
            testbed.nerf.training.train_mode = ngp.TrainMode.Rfl
        elif (
            args.train_mode.lower() == "rfl_relax"
            or args.train_mode.lower() == "rflrelax"
        ):
            testbed.nerf.training.train_mode = ngp.TrainMode.RflRelax
        else:
            raise ValueError(f"Unknown train mode: {args.train_mode}")

    if args.nerf_compatibility:
        print("NeRF compatibility mode enabled")

        testbed.color_space = ngp.ColorSpace.SRGB
        testbed.nerf.cone_angle_constant = 0
        testbed.nerf.training.random_bg_color = False

        if testbed.nerf.training.train_mode != ngp.TrainMode.Nerf:
            print(
                f"Warning: forcing train mode to NeRF for nerf compatibility "
                f"(was {testbed.nerf.training.train_mode})"
            )
        testbed.nerf.training.train_mode = ngp.TrainMode.Nerf

    old_training_step = 0
    n_steps = args.n_steps

    if n_steps < 0 and (not args.load_snapshot or args.gui):
        n_steps = 35000

    original_train_mode = ngp.TrainMode(testbed.nerf.training.train_mode)
    prev_train_mode = original_train_mode
    use_training_schedule = True

    tqdm_last_update = 0
    last_val_step = 0
    train_dataset_path = args.scene

    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="steps") as t:
            while testbed.frame():
                if (
                    prev_train_mode != testbed.nerf.training.train_mode
                    and use_training_schedule
                ):
                    print(
                        "Disabling Rfl/RflRelax training schedule due to UI train mode change"
                    )
                    use_training_schedule = False

                if testbed.want_repl():
                    repl(testbed)

                if testbed.training_step >= n_steps:
                    if args.gui:
                        testbed.shall_train = False
                    else:
                        break

                if use_training_schedule:
                    if original_train_mode == ngp.TrainMode.RflRelax:
                        if (
                            args.rflrelax_begin_step
                            <= testbed.training_step
                            < args.rflrelax_end_step
                        ):
                            testbed.nerf.training.train_mode = ngp.TrainMode.RflRelax
                        else:
                            testbed.nerf.training.train_mode = ngp.TrainMode.Nerf
                    elif original_train_mode == ngp.TrainMode.Rfl:
                        if testbed.training_step > args.rfl_warmup_steps:
                            testbed.nerf.training.train_mode = ngp.TrainMode.Rfl
                        else:
                            testbed.nerf.training.train_mode = ngp.TrainMode.Nerf

                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    step = int(testbed.training_step)
                    t.update(step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = step
                    tqdm_last_update = now

                    if wb_run is not None and HAS_WANDB:
                        wb_run.log(
                            {
                                "train/loss": float(testbed.loss),
                                "train/step": step,
                            },
                            step=step,
                        )

                    if (
                        not args.gui
                        and args.val_interval > 0
                        and step >= last_val_step + args.val_interval
                    ):
                        print(f"[VAL] Running validation at step {step}...")

                        old_bg = testbed.background_color
                        old_snap = testbed.snap_to_pixel_centers
                        old_render_min = testbed.nerf.render_min_transmittance
                        old_shall_train = testbed.shall_train

                        testbed.background_color = [0.0, 0.0, 0.0, 1.0]
                        testbed.shall_train = False

                        evaluate_current_dataset(
                            testbed,
                            split_name="train",
                            max_images=1, # For train log just one image
                            log_first_images=args.val_log_images,
                            wb_run=wb_run,
                            step=step,
                            enable_face_crop=args.face_crop,
                            face_cascade=face_cascade,
                            save_prefix="val_train",
                        )

                        if args.test_transforms:
                            print(
                                f"[VAL] Evaluating test transforms from {args.test_transforms}"
                            )
                            testbed.load_training_data(args.test_transforms)
                            evaluate_current_dataset(
                                testbed,
                                split_name="test",
                                max_images=args.val_max_images,
                                log_first_images=args.val_log_images,
                                wb_run=wb_run,
                                step=step,
                                enable_face_crop=args.face_crop,
                                face_cascade=face_cascade,
                                save_prefix="val_test",
                            )
                            if train_dataset_path:
                                testbed.load_training_data(train_dataset_path)

                        testbed.background_color = old_bg
                        testbed.snap_to_pixel_centers = old_snap
                        testbed.nerf.render_min_transmittance = old_render_min
                        testbed.shall_train = old_shall_train

                        last_val_step = step

                prev_train_mode = ngp.TrainMode(testbed.nerf.training.train_mode)

    if args.save_snapshot:
        os.makedirs(os.path.dirname(args.save_snapshot), exist_ok=True)
        testbed.save_snapshot(args.save_snapshot, False)

    if not args.gui:
        print("[FINAL VAL] Training split.")
        testbed.background_color = [0.0, 0.0, 0.0, 1.0]
        evaluate_current_dataset(
            testbed,
            split_name="train",
            max_images=args.val_max_images,
            log_first_images=True,
            wb_run=wb_run,
            step=int(testbed.training_step),
            enable_face_crop=args.face_crop,
            face_cascade=face_cascade,
            save_prefix="final_train",
        )

        if args.test_transforms:
            print(
                f"[FINAL VAL] Evaluating test transforms from {args.test_transforms}"
            )
            testbed.load_training_data(args.test_transforms)
            evaluate_current_dataset(
                testbed,
                split_name="test",
                max_images=args.val_max_images,
                log_first_images=True,
                wb_run=wb_run,
                step=int(testbed.training_step),
                enable_face_crop=args.face_crop,
                face_cascade=face_cascade,
                save_prefix="final_test",
            )
            if args.scene:
                testbed.load_training_data(args.scene)

    if args.save_mesh:
        res = args.marching_cubes_res or 256
        thresh = args.marching_cubes_density_thresh or 2.5
        print(
            f"Generating mesh via marching cubes and saving to {args.save_mesh}. "
            f"Resolution=[{res},{res},{res}], Density Threshold={thresh}"
        )
        testbed.compute_and_save_marching_cubes_mesh(
            args.save_mesh, [res, res, res], thresh=thresh
        )

    if "ref_transforms" in locals() and ref_transforms:
        testbed.fov_axis = 0
        testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
        if not args.screenshot_frames:
            args.screenshot_frames = range(len(ref_transforms["frames"]))
        print(args.screenshot_frames)
        for idx in args.screenshot_frames:
            f = ref_transforms["frames"][int(idx)]

            if "transform_matrix" in f:
                cam_matrix = f["transform_matrix"]
            elif "transform_matrix_start" in f:
                cam_matrix = f["transform_matrix_start"]
            else:
                raise KeyError(
                    "Missing both 'transform_matrix' and 'transform_matrix_start'"
                )

            testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])
            outname = os.path.join(
                args.screenshot_dir, os.path.basename(f["file_path"])
            )

            if not os.path.splitext(outname)[1]:
                outname = outname + ".png"

            print(f"rendering {outname}")
            image = testbed.render(
                args.width or int(ref_transforms["w"]),
                args.height or int(ref_transforms["h"]),
                args.screenshot_spp,
                True,
            )
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            write_image(outname, image)
    elif args.screenshot_dir:
        outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
        print(f"Rendering {outname}.png")
        image = testbed.render(
            args.width or 1920, args.height or 1080, args.screenshot_spp, True
        )
        if os.path.dirname(outname) != "":
            os.makedirs(os.path.dirname(outname), exist_ok=True)
        write_image(outname + ".png", image)

    if args.aabb:
        min_x, min_y, min_z, max_x, max_y, max_z = args.aabb
        testbed.render_aabb = ngp.BoundingBox(
            [min_x, min_y, min_z], [max_x, max_y, max_z]
        )
        testbed.render_aabb_to_local = np.eye(3)

    if args.video_camera_path:
        testbed.load_camera_path(args.video_camera_path)

        resolution = [args.width or 1920, args.height or 1080]
        save_frames = "%" in args.video_output

        start_frame, end_frame = args.video_render_range
        n_frames = args.video_n_seconds * args.video_fps

        if "tmp" in os.listdir():
            shutil.rmtree("tmp")
        os.makedirs("tmp")

        for i in tqdm(
            list(range(0, end_frame)), unit="frames", desc="Rendering video"
        ):
            testbed.camera_smoothing = args.video_camera_smoothing

            if start_frame >= 0 and i < start_frame:
                _ = testbed.render(
                    32,
                    32,
                    1,
                    True,
                    float(i) / n_frames,
                    float(i + 1) / n_frames,
                    args.video_fps,
                    shutter_fraction=0.5,
                )
                continue
            elif end_frame >= 0 and i > end_frame:
                continue

            frame = testbed.render(
                resolution[0],
                resolution[1],
                args.video_spp,
                True,
                float(i) / n_frames,
                float(i + 1) / n_frames,
                args.video_fps,
                shutter_fraction=0.5,
            )
            if save_frames:
                write_image(
                    args.video_output % i,
                    np.clip(frame * 2 ** args.exposure, 0.0, 1.0),
                    quality=30,
                )
            else:
                write_image(
                    f"tmp/{i:04d}.jpg",
                    np.clip(frame * 2 ** args.exposure, 0.0, 1.0),
                    quality=100,
                )

        if not save_frames:
            os.system(
                f"{ffmpeg_path} -y -framerate {args.video_fps} -i tmp/%04d.jpg "
                f"-c:v libx264 -pix_fmt yuv420p {args.video_output}"
            )

        if "tmp" in os.listdir():
            shutil.rmtree("tmp")

    if wb_run is not None and HAS_WANDB:
        wb_run.finish()