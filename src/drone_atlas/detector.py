"""Model loading, YOLO output decoding, NMS, and annotation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Any

try:
    import cv2 as _cv2
except ImportError:  # Tests can exercise pure helpers without OpenCV installed.
    _cv2 = None


class ModelError(RuntimeError):
    """Raised when model assets cannot be loaded or used."""


@dataclass(frozen=True)
class ModelPaths:
    """Filesystem locations for a Darknet YOLO model and its labels."""

    config: Path
    weights: Path
    labels: Path

    @classmethod
    def from_directory(
        cls,
        model_dir: Path | str,
        *,
        config: Path | str | None = None,
        weights: Path | str | None = None,
        labels: Path | str | None = None,
    ) -> "ModelPaths":
        directory = Path(model_dir)
        return cls(
            config=Path(config) if config else directory / "yolov3.cfg",
            weights=Path(weights) if weights else directory / "yolov3.weights",
            labels=Path(labels) if labels else directory / "coco.names",
        )

    def validate(self) -> None:
        missing = [path for path in (self.config, self.weights, self.labels) if not path.is_file()]
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise ModelError(
                f"Missing model asset(s): {rendered}. "
                "Run scripts/download_models.py --model yolov3 or provide explicit paths."
            )


@dataclass(frozen=True)
class Detection:
    """One decoded detection using an integer (x, y, width, height) box."""

    class_id: int
    confidence: float
    box: tuple[int, int, int, int]


def _require_cv2() -> Any:
    if _cv2 is None:
        raise ModelError(
            "OpenCV is not installed. Install the project with `pip install -e .`."
        )
    return _cv2


def _validate_threshold(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")


def load_labels(path: Path) -> tuple[str, ...]:
    try:
        labels = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError as exc:
        raise ModelError(f"Could not read labels from {path}: {exc}") from exc
    if not labels:
        raise ModelError(f"Label file is empty: {path}")
    return labels


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return a stable, high-contrast BGR colour for a class identifier."""

    digest = hashlib.sha256(f"drone-atlas:{class_id}".encode("ascii")).digest()
    return tuple(64 + component % 192 for component in digest[:3])


def normalize_nms_indices(indices: Any) -> list[int]:
    """Normalize the index shapes returned by different OpenCV releases."""

    if indices is None:
        return []
    if hasattr(indices, "tolist"):
        indices = indices.tolist()

    flattened: list[int] = []

    def visit(value: Any) -> None:
        if isinstance(value, Integral) and not isinstance(value, bool):
            index = int(value)
            if index not in flattened:
                flattened.append(index)
            return
        if isinstance(value, (str, bytes)):
            return
        if isinstance(value, Iterable):
            for item in value:
                visit(item)

    visit(indices)
    return flattened


def decode_yolo_outputs(
    outputs: Sequence[Any],
    image_shape: Sequence[int],
    confidence_threshold: float,
) -> list[Detection]:
    """Decode normalized YOLOv3 rows into clipped image-space detections."""

    _validate_threshold(confidence_threshold, "confidence threshold")
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")

    detections: list[Detection] = []
    for output in outputs:
        for row in output:
            if len(row) < 6:
                continue
            class_scores = row[5:]
            if len(class_scores) == 0:
                continue
            class_id = max(range(len(class_scores)), key=lambda index: float(class_scores[index]))
            confidence = float(row[4]) * float(class_scores[class_id])
            if confidence < confidence_threshold:
                continue

            centre_x = float(row[0]) * image_width
            centre_y = float(row[1]) * image_height
            box_width = max(0, round(float(row[2]) * image_width))
            box_height = max(0, round(float(row[3]) * image_height))
            raw_left = round(centre_x - box_width / 2)
            raw_top = round(centre_y - box_height / 2)
            left = max(0, raw_left)
            top = max(0, raw_top)
            right = min(image_width, raw_left + box_width)
            bottom = min(image_height, raw_top + box_height)
            clipped_width = right - left
            clipped_height = bottom - top
            if clipped_width <= 0 or clipped_height <= 0:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    confidence=confidence,
                    box=(left, top, clipped_width, clipped_height),
                )
            )
    return detections


def _apply_class_aware_nms(
    detections: Sequence[Detection],
    confidence_threshold: float,
    nms_threshold: float,
    cv_backend: Any,
) -> list[Detection]:
    kept: list[Detection] = []
    class_ids = sorted({detection.class_id for detection in detections})
    for class_id in class_ids:
        candidates = [detection for detection in detections if detection.class_id == class_id]
        boxes = [list(detection.box) for detection in candidates]
        scores = [detection.confidence for detection in candidates]
        raw_indices = cv_backend.dnn.NMSBoxes(
            boxes,
            scores,
            confidence_threshold,
            nms_threshold,
        )
        for index in normalize_nms_indices(raw_indices):
            if 0 <= index < len(candidates):
                kept.append(candidates[index])
    return sorted(kept, key=lambda detection: detection.confidence, reverse=True)


class YoloV3Detector:
    """Headless YOLOv3 detector backed by OpenCV DNN."""

    def __init__(
        self,
        paths: ModelPaths,
        *,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (416, 416),
        cv_backend: Any | None = None,
    ) -> None:
        _validate_threshold(confidence_threshold, "confidence threshold")
        _validate_threshold(nms_threshold, "NMS threshold")
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError("input_size dimensions must be positive")
        paths.validate()

        self.paths = paths
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.cv = cv_backend or _require_cv2()
        self.labels = load_labels(paths.labels)
        try:
            self.net = self.cv.dnn.readNetFromDarknet(str(paths.config), str(paths.weights))
            self.output_layer_names = self._resolve_output_layers()
        except Exception as exc:
            raise ModelError(f"Could not load YOLOv3 model: {exc}") from exc

    def _resolve_output_layers(self) -> list[str]:
        if hasattr(self.net, "getUnconnectedOutLayersNames"):
            return list(self.net.getUnconnectedOutLayersNames())
        layer_names = self.net.getLayerNames()
        layer_ids = normalize_nms_indices(self.net.getUnconnectedOutLayers())
        return [layer_names[index - 1] for index in layer_ids if 1 <= index <= len(layer_names)]

    def detect(self, image: Any) -> list[Detection]:
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            raise ValueError("image must be a non-empty OpenCV image")
        blob = self.cv.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layer_names)
        candidates = decode_yolo_outputs(outputs, image.shape, self.confidence_threshold)
        return _apply_class_aware_nms(
            candidates,
            self.confidence_threshold,
            self.nms_threshold,
            self.cv,
        )

    def annotate(self, image: Any, detections: Sequence[Detection]) -> Any:
        annotated = image.copy()
        for detection in detections:
            x, y, width, height = detection.box
            colour = class_color(detection.class_id)
            label_name = (
                self.labels[detection.class_id]
                if 0 <= detection.class_id < len(self.labels)
                else f"class-{detection.class_id}"
            )
            text = f"{label_name} {detection.confidence:.2f}"
            self.cv.rectangle(annotated, (x, y), (x + width, y + height), colour, 2)
            text_y = max(16, y - 6)
            self.cv.putText(
                annotated,
                text,
                (x, text_y),
                self.cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                2,
                self.cv.LINE_AA,
            )
        return annotated
