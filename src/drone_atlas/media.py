"""Headless still-image and video processing workflows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .detector import YoloV3Detector


class MediaError(RuntimeError):
    """Raised when an input cannot be decoded or an output cannot be written."""


@dataclass(frozen=True)
class MediaResult:
    frames_processed: int
    detections: int
    output: Path


def _validate_input(path: Path) -> None:
    if not path.is_file():
        raise MediaError(f"Input file does not exist: {path}")


def _prepare_output(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise MediaError("Input and output paths must be different")
    output_path.parent.mkdir(parents=True, exist_ok=True)


def process_image(
    input_path: Path | str,
    output_path: Path | str,
    detector: YoloV3Detector,
    *,
    cv_backend: Any | None = None,
) -> MediaResult:
    source, destination = Path(input_path), Path(output_path)
    _validate_input(source)
    _prepare_output(source, destination)
    cv = cv_backend or detector.cv
    image = cv.imread(str(source))
    if image is None:
        raise MediaError(f"OpenCV could not decode image: {source}")
    detections = detector.detect(image)
    annotated = detector.annotate(image, detections)
    if not cv.imwrite(str(destination), annotated):
        raise MediaError(f"OpenCV could not write image: {destination}")
    return MediaResult(frames_processed=1, detections=len(detections), output=destination)


def process_video(
    input_path: Path | str,
    output_path: Path | str,
    detector: YoloV3Detector,
    *,
    cv_backend: Any | None = None,
) -> MediaResult:
    source, destination = Path(input_path), Path(output_path)
    _validate_input(source)
    _prepare_output(source, destination)
    cv = cv_backend or detector.cv
    capture = cv.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise MediaError(f"OpenCV could not open video: {source}")

    writer = None
    frames_processed = 0
    detection_count = 0
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise MediaError(f"Video contains no readable frames: {source}")
        height, width = frame.shape[:2]
        fps = float(capture.get(cv.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        codec = "mp4v" if destination.suffix.lower() in {".mp4", ".m4v", ".mov"} else "XVID"
        writer = cv.VideoWriter(
            str(destination),
            cv.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise MediaError(f"OpenCV could not create video writer: {destination}")

        while ok and frame is not None:
            detections = detector.detect(frame)
            writer.write(detector.annotate(frame, detections))
            frames_processed += 1
            detection_count += len(detections)
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return MediaResult(
        frames_processed=frames_processed,
        detections=detection_count,
        output=destination,
    )
