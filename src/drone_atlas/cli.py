"""Command-line interface for DroneAtlas."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

from .detector import ModelError, ModelPaths, YoloV3Detector
from .media import MediaError, process_image, process_video


def _threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 inclusive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone-atlas",
        description="Run headless YOLOv3 object detection on an image or video.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("image", "video"):
        subparser = subparsers.add_parser(command, help=f"process one {command}")
        subparser.add_argument("--input", required=True, type=Path, help="input media path")
        subparser.add_argument("--output", required=True, type=Path, help="annotated output path")
        subparser.add_argument("--model-dir", type=Path, default=Path("models"))
        subparser.add_argument("--config", type=Path, help="override Darknet .cfg path")
        subparser.add_argument("--weights", type=Path, help="override Darknet .weights path")
        subparser.add_argument("--labels", type=Path, help="override class-label path")
        subparser.add_argument("--confidence", type=_threshold, default=0.5)
        subparser.add_argument("--nms", type=_threshold, default=0.4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")

    paths = ModelPaths.from_directory(
        args.model_dir,
        config=args.config,
        weights=args.weights,
        labels=args.labels,
    )
    try:
        detector = YoloV3Detector(
            paths,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms,
        )
        runner = process_image if args.command == "image" else process_video
        result = runner(args.input, args.output, detector)
    except (MediaError, ModelError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Processed {result.frames_processed} frame(s), found "
        f"{result.detections} detection(s), wrote {result.output}"
    )
    return 0
