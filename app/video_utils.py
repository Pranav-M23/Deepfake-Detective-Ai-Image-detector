"""Utilities for extracting and preprocessing video frames."""

from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    duration = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    info["duration_seconds"] = round(duration, 2)
    cap.release()
    return info


def sample_frames(
    video_path: str,
    every_n: int = 15,
    max_frames: int = 20,
) -> List[np.ndarray]:
    """
    Sample up to `max_frames` BGR frames from a video.

    Uses *uniform* spacing across the full video so that short and long
    videos are both covered evenly, rather than only sampling the first
    portion of the video.

    Args:
        video_path: Path to the video file.
        every_n:    Fallback stride when the video is too short to
                    use uniform spacing.  Ignored when total_frames is
                    large enough to auto-compute a stride.
        max_frames: Maximum number of frames to return.

    Returns:
        List of BGR numpy arrays (HxWx3 uint8).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute uniform sample positions
    if total > 0:
        n = min(max_frames, total)
        if n <= 1:
            positions = [0]
        else:
            positions = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
    else:
        # Unknown length — fall back to stride-based sampling
        positions = None

    frames: List[np.ndarray] = []

    if positions is not None:
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        # Stream-based fallback
        idx = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every_n == 0:
                frames.append(frame)
            idx += 1

    cap.release()
    return frames


def resize_frame(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize a BGR frame to (size × size)."""
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
