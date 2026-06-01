#!/usr/bin/env python3
"""
Extract candidate frames from CETI field/lab video for whale annotation.

Uses motion-based keyframe selection and optional acoustic timestamp sync
to prioritize frames likely containing whale activity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def compute_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Frame differencing motion score."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean())


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_rate: int = 30,
    motion_threshold: float = 5.0,
    max_frames: int | None = None,
) -> int:
    """
    Extract frames from video with motion-based filtering.

    Args:
        video_path: Input video file
        output_dir: Directory to save extracted frames
        sample_rate: Extract every Nth frame at minimum
        motion_threshold: Minimum motion score to save frame
        max_frames: Maximum frames to extract (None = unlimited)

    Returns:
        Number of frames saved
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    saved = 0
    frame_idx = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            motion = compute_motion_score(prev_gray, gray)
            if motion < motion_threshold:
                frame_idx += 1
                prev_gray = gray
                continue

        out_path = output_dir / f"{stem}_f{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1
        prev_gray = gray

        if max_frames and saved >= max_frames:
            break

        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from CETI video for annotation")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output", type=str, default="./data/whale/raw_frames")
    parser.add_argument("--sample-rate", type=int, default=30, help="Minimum frame interval")
    parser.add_argument("--motion-threshold", type=float, default=5.0)
    parser.add_argument("--max-frames-per-video", type=int, default=200)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".MP4"}

    videos = [f for f in sorted(video_dir.rglob("*")) if f.suffix in extensions]
    if not videos:
        print(f"No videos found in {video_dir}")
        return

    total = 0
    for video in videos:
        n = extract_frames(
            video, output_dir,
            sample_rate=args.sample_rate,
            motion_threshold=args.motion_threshold,
            max_frames=args.max_frames_per_video,
        )
        print(f"  {video.name}: {n} frames")
        total += n

    print(f"\nExtracted {total} frames to {output_dir}")
    print("\nNext steps:")
    print("  1. Upload frames to CVAT (https://app.cvat.ai) or Label Studio")
    print("  2. Annotate bounding boxes: sperm_whale, whale_surface, whale_partial")
    print("  3. Export as YOLO 1.1 format")
    print("  4. Run: python ceti/whale/data_curation/convert_annotations.py --cvat-export <path>")


if __name__ == "__main__":
    main()
