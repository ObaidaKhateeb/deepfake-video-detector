"""
core/video_loader.py
Loads a video file and provides frames + basic metadata.
"""

import cv2
import os
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class VideoData:
    path: str
    frames: List[np.ndarray] = field(default_factory=list)
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    duration_sec: float = 0.0
    file_size_mb: float = 0.0
    has_audio: bool = False  # basic flag; deep audio needs ffprobe


def load_video(path: str, max_frames: int = 120, sample_rate: int = 5) -> VideoData:
    """
    Load a video file and extract sampled frames.

    Args:
        path:        Absolute path to the video file.
        max_frames:  Hard cap on frames to keep in memory.
        sample_rate: Extract every Nth frame (1 = every frame).

    Returns:
        VideoData with frames list and metadata populated.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = frame_count / fps if fps > 0 else 0.0
    file_size   = os.path.getsize(path) / (1024 * 1024)

    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frames.append(frame)
        idx += 1

    cap.release()

    return VideoData(
        path=path,
        frames=frames,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=duration,
        file_size_mb=file_size,
    )
