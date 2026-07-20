"""Reusable OpenCV/YOLOv3 detection components."""

from .detector import (
    Detection,
    ModelError,
    ModelPaths,
    YoloV3Detector,
    class_color,
    decode_yolo_outputs,
    normalize_nms_indices,
)
from .media import MediaError, MediaResult, process_image, process_video

__all__ = [
    "Detection",
    "MediaError",
    "MediaResult",
    "ModelError",
    "ModelPaths",
    "YoloV3Detector",
    "class_color",
    "decode_yolo_outputs",
    "normalize_nms_indices",
    "process_image",
    "process_video",
]
