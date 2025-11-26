#!/usr/bin/env python3
"""
YOLO Person Detection and Cropping Script with SAM 2.1 Segmentation
Processes images recursively, detects persons, crops with padding, and optionally applies SAM 2.1 masking.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from tqdm import tqdm


def get_image_files(input_dir: str, extensions: List[str] = None) -> List[Path]:
    """Recursively find all image files in the directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']

    input_path = Path(input_dir)
    image_files = []

    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))

    return sorted(set(image_files))


def calculate_padding(bbox_w: int, bbox_h: int, padding_percent: float) -> Tuple[int, int]:
    """Calculate padding based on bbox size (square root of area)"""
    bbox_area = bbox_w * bbox_h
    padding_base = int(np.sqrt(bbox_area) * (padding_percent / 100.0))
    return padding_base, padding_base


def expand_bbox(x1: int, y1: int, x2: int, y2: int,
                pad_x: int, pad_y: int,
                img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Expand bounding box with padding while keeping within image bounds"""
    x1_expanded = max(0, x1 - pad_x)
    y1_expanded = max(0, y1 - pad_y)
    x2_expanded = min(img_w, x2 + pad_x)
    y2_expanded = min(img_h, y2 + pad_y)
    return x1_expanded, y1_expanded, x2_expanded, y2_expanded


def apply_sam2_mask(image: np.ndarray,
                    bbox_xyxy: Tuple[int, int, int, int],
                    sam_model: SAM,
                    background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Apply SAM 2.1 segmentation using the YOLO person bbox as a prompt and mask out the background.
    We run SAM on the full image with a bbox prompt, then build a background-filled result.
    """
    x1, y1, x2, y2 = bbox_xyxy

    # Run SAM 2.1 with the bbox prompt (XYXY format)
    # Ultralytics SAM API accepts bboxes in image coordinates.
    results = sam_model(image, bboxes=[[x1, y1, x2, y2]])

    if not results or results[0].masks is None:
        print("Warning: SAM2.1 returned no masks for the provided bbox.")
        return image

    # Combine all masks (in case multiple instance proposals returned)
    masks = results[0].masks.data  # torch.Tensor [N, H, W]
    # Move to CPU numpy uint8
    combined_mask = np.any(masks.cpu().numpy() > 0.5, axis=0).astype(np.uint8)

    # Create 3-channel mask
    mask_3ch = np.stack([combined_mask] * 3, axis=-1)

    # Compose background
    background = np.full_like(image, background_color, dtype=np.uint8)

    # Keep foreground where mask=1, else paint background
    result = np.where(mask_3ch, image, background)
    return result


def process_image(image_path: Path,
                  yolo_model: YOLO,
                  input_dir: Path,
                  output_dir: Path,
                  padding_percent: float,
                  sam_model: SAM = None,
                  background_color: Tuple[int, int, int] = (255, 255, 255),
                  confidence_threshold: float = 0.25) -> int:
    """Process a single image: detect persons, pick the highest-confidence one, crop with padding, optionally apply SAM 2.1"""
    try:
        relative_path = image_path.relative_to(input_dir)
        output_subdir = output_dir / relative_path.parent
    except ValueError:
        output_subdir = output_dir
    output_subdir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read {image_path}")
        return 0

    img_h, img_w = image.shape[:2]
    results = yolo_model(image, verbose=False)

    candidates = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.squeeze().cpu().item())
            conf = float(box.conf.squeeze().cpu().item())
            if cls_id == 0 and conf >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy.squeeze().cpu().numpy())
                bbox_w, bbox_h = x2 - x1, y2 - y1
                area = max(0, bbox_w) * max(0, bbox_h)
                candidates.append({"conf": conf, "xyxy": (x1, y1, x2, y2), "area": area})

    if not candidates:
        return 0

    best = max(candidates, key=lambda c: (c["conf"], c["area"]))
    x1, y1, x2, y2 = best["xyxy"]
    bbox_w, bbox_h = x2 - x1, y2 - y1

    pad_x, pad_y = calculate_padding(bbox_w, bbox_h, padding_percent)
    x1_exp, y1_exp, x2_exp, y2_exp = expand_bbox(x1, y1, x2, y2, pad_x, pad_y, img_w, img_h)

    if sam_model is not None:
        masked_full = apply_sam2_mask(image, (x1, y1, x2, y2), sam_model, background_color)
        crop = masked_full[y1_exp:y2_exp, x1_exp:x2_exp]
    else:
        crop = image[y1_exp:y2_exp, x1_exp:x2_exp]

    # сохраняем под тем же именем, что и исходный файл
    output_path = output_subdir / image_path.name
    ext = image_path.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif ext == '.png':
        cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(str(output_path), crop)

    return 1



import shutil

def main():
    parser = argparse.ArgumentParser(
        description='Detect persons in images using YOLO v11 and create cropped images (optional SAM 2.1 refinement).'
    )
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument('output_dir', type=str, help='Output directory for cropped images')
    parser.add_argument('--padding', type=float, default=25.0,
                        help='Padding percentage based on sqrt(bbox_area) (default: 25%%)')
    parser.add_argument('--confidence', type=float, default=0.25,
                        help='Confidence threshold for detection (default: 0.25)')
    parser.add_argument('--yolo-model', type=str, default='yolo11x.pt',
                        help='YOLO model to use (default: yolo11x.pt)')
    parser.add_argument('--use-sam', action='store_true',
                        help='Enable SAM 2.1 segmentation guided by the YOLO bbox')
    parser.add_argument('--sam-model', type=str, default='sam2.1_l.pt',
                        help='SAM 2.1 model to use (default: sam2.1_l.pt)')
    parser.add_argument('--bg-color', type=str, default='255,255,255',
                        help='Background color for mask in BGR format, e.g. 255,255,255')

    args = parser.parse_args()

    # Parse background color
    try:
        bg_color = tuple(map(int, args.bg_color.split(',')))
        if len(bg_color) != 3:
            raise ValueError
    except Exception:
        print("Error: Background color must be in format: B,G,R (e.g., 255,255,255)")
        return

    # Prepare output directory: delete if exists
    output_path = Path(args.output_dir)
    if output_path.exists():
        print(f"⚠️  Output directory already exists, deleting: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load YOLO
    print(f"Loading YOLO model: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)

    # Load SAM 2.1 if requested
    sam_model = None
    if args.use_sam:
        print(f"Loading SAM 2.1 model: {args.sam_model}")
        sam_model = SAM(args.sam_model)
        print(f"Background color (BGR): {bg_color}")

    # Scan images
    print(f"Scanning for images in: {args.input_dir}")
    image_files = get_image_files(args.input_dir)
    print(f"Found {len(image_files)} images")

    if not image_files:
        print("No images found!")
        return

    # Process images
    total_crops = 0
    input_path = Path(args.input_dir)

    for image_path in tqdm(image_files, desc="Processing images"):
        crops = process_image(
            image_path,
            yolo_model,
            input_path,
            output_path,
            args.padding,
            sam_model,
            bg_color,
            args.confidence
        )
        total_crops += crops

    print(f"\nProcessing complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total person crops saved: {total_crops}")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    main()
