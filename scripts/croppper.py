#!/usr/bin/env python3
"""
YOLO Person Detection and Cropping Script with SAM3 Segmentation
Processes images recursively, detects persons, crops with padding, and optionally applies SAM3 masking
"""

import argparse
import os
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


def apply_sam3_mask(image: np.ndarray,
					bbox: Tuple[int, int, int, int],
					sam_model: SAM,
					text_prompt: str,
					background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
	"""Apply SAM3 segmentation with text prompt and mask out background"""
	x1, y1, x2, y2 = bbox

	# Crop the region for SAM3 processing
	crop = image[y1:y2, x1:x2].copy()

	# Run SAM3 with text prompt
	results = sam_model(crop, texts=text_prompt)

	if len(results) > 0 and results[0].masks is not None:
		# Get the mask
		masks = results[0].masks.data.cpu().numpy()

		# Combine all masks (in case multiple objects detected)
		combined_mask = np.any(masks, axis=0).astype(np.uint8)

		# Resize mask to match crop size if needed
		if combined_mask.shape != crop.shape[:2]:
			combined_mask = cv2.resize(combined_mask,
									   (crop.shape[1], crop.shape[0]),
									   interpolation=cv2.INTER_NEAREST)

		# Create 3-channel mask
		mask_3ch = np.stack([combined_mask] * 3, axis=-1)

		# Create background
		background = np.full_like(crop, background_color, dtype=np.uint8)

		# Apply mask: keep masked regions, fill rest with background
		masked_crop = np.where(mask_3ch, crop, background)

		# Create full image with background
		result = np.full_like(image, background_color, dtype=np.uint8)
		result[y1:y2, x1:x2] = masked_crop

		return result
	else:
		# If no mask found, return original image
		print(f"Warning: No mask found for prompt '{text_prompt}'")
		return image


def process_image(image_path: Path,
				  yolo_model: YOLO,
				  input_dir: Path,
				  output_dir: Path,
				  padding_percent: float,
				  sam_model: SAM = None,
				  sam_prompt: str = None,
				  background_color: Tuple[int, int, int] = (255, 255, 255),
				  confidence_threshold: float = 0.25) -> int:
	"""Process a single image: detect persons, crop with padding, optionally apply SAM3"""

	# Calculate relative path from input directory to preserve folder structure
	try:
		relative_path = image_path.relative_to(input_dir)
		output_subdir = output_dir / relative_path.parent
	except ValueError:
		# If image_path is not relative to input_dir, use direct output
		output_subdir = output_dir

	# Create output subdirectory structure
	output_subdir.mkdir(parents=True, exist_ok=True)

	# Read image
	image = cv2.imread(str(image_path))
	if image is None:
		print(f"Error: Could not read {image_path}")
		return 0

	img_h, img_w = image.shape[:2]

	# Run YOLO detection
	results = yolo_model(image, verbose=False)

	crops_saved = 0

	# Process detections
	for result in results:
		boxes = result.boxes

		for i, box in enumerate(boxes):
			# Check if detection is a person (class 0 in COCO dataset)
			if int(box.cls) == 0 and float(box.conf) >= confidence_threshold:
				# Get bounding box coordinates
				x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

				bbox_w = x2 - x1
				bbox_h = y2 - y1

				# Calculate padding
				pad_x, pad_y = calculate_padding(bbox_w, bbox_h, padding_percent)

				# Expand bbox with padding
				x1_exp, y1_exp, x2_exp, y2_exp = expand_bbox(
					x1, y1, x2, y2, pad_x, pad_y, img_w, img_h
				)

				# Apply SAM3 if requested
				if sam_model is not None and sam_prompt:
					processed_image = apply_sam3_mask(
						image,
						(x1_exp, y1_exp, x2_exp, y2_exp),
						sam_model,
						sam_prompt,
						background_color
					)
					crop = processed_image[y1_exp:y2_exp, x1_exp:x2_exp]
				else:
					# Crop the expanded region
					crop = image[y1_exp:y2_exp, x1_exp:x2_exp]

				# Create output filename
				stem = image_path.stem
				ext = image_path.suffix
				output_filename = f"{stem}_person_{i:03d}{ext}"
				output_path = output_subdir / output_filename

				# Save crop in original resolution with maximum quality
				# Set compression parameters based on file format
				if ext.lower() in ['.jpg', '.jpeg']:
					# JPEG: quality 100 (maximum)
					cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
				elif ext.lower() == '.png':
					# PNG: compression level 0 (no compression, fastest, largest file)
					# For best quality with reasonable file size, use level 3
					cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])
				elif ext.lower() in ['.tiff', '.tif']:
					# TIFF: no compression
					cv2.imwrite(str(output_path), crop)
				else:
					cv2.imwrite(str(output_path), crop)
				crops_saved += 1

	return crops_saved


def main():
	parser = argparse.ArgumentParser(
		description='Detect persons in images using YOLO v11 and create cropped images'
	)
	parser.add_argument('input_dir', type=str, help='Input directory containing images')
	parser.add_argument('output_dir', type=str, help='Output directory for cropped images')
	parser.add_argument('--padding', type=float, default=25.0,
						help='Padding percentage based on sqrt(bbox_area) (default: 20%%)')
	parser.add_argument('--confidence', type=float, default=0.25,
						help='Confidence threshold for detection (default: 0.25)')
	parser.add_argument('--yolo-model', type=str, default='yolo11x.pt',
						help='YOLO model to use (default: yolo11x.pt - largest model)')
	parser.add_argument('--use-sam3', action='store_true',
						help='Enable SAM3 segmentation with text prompt')
	parser.add_argument('--sam-prompt', type=str, default='person',
						help='Text prompt for SAM3 segmentation (default: person)')
	parser.add_argument('--sam-model', type=str, default='sam2.1_l.pt',
						help='SAM3 model to use (default: sam3-l.pt - largest model)')
	parser.add_argument('--bg-color', type=str, default='255,255,255',
						help='Background color for SAM3 mask in BGR format (default: 255,255,255 - white)')

	args = parser.parse_args()

	# Parse background color
	bg_color = tuple(map(int, args.bg_color.split(',')))
	if len(bg_color) != 3:
		print("Error: Background color must be in format: B,G,R (e.g., 255,255,255)")
		return

	# Create output directory
	output_path = Path(args.output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	# Load YOLO model
	print(f"Loading YOLO model: {args.yolo_model}")
	yolo_model = YOLO(args.yolo_model)

	# Load SAM3 model if requested
	sam_model = None
	if args.use_sam3:
		print(f"Loading SAM3 model: {args.sam_model}")
		sam_model = SAM(args.sam_model)
		print(f"SAM3 text prompt: '{args.sam_prompt}'")
		print(f"Background color (BGR): {bg_color}")

	# Get all image files
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
			args.sam_prompt if args.use_sam3 else None,
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
