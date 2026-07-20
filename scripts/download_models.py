"""Download official YOLOv3 assets without placing them under version control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import urllib.error
import urllib.request


DARKNET_COMMIT = "f6afaabcdf85f77e7aff2ec55c020c0e297c77f9"


@dataclass(frozen=True)
class Asset:
    filename: str
    url: str
    sha256: str | None


YOLOV3_ASSETS = (
    Asset(
        "yolov3.cfg",
        f"https://raw.githubusercontent.com/pjreddie/darknet/{DARKNET_COMMIT}/cfg/yolov3.cfg",
        "22489ea38575dfa36c67a90048e8759576416a79d32dc11e15d2217777b9a953",
    ),
    Asset(
        "coco.names",
        f"https://raw.githubusercontent.com/pjreddie/darknet/{DARKNET_COMMIT}/data/coco.names",
        "634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84",
    ),
    Asset(
        "yolov3.weights",
        "https://pjreddie.com/media/files/yolov3.weights",
        "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: Asset, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        digest = sha256_file(destination)
        if asset.sha256 and digest != asset.sha256:
            raise RuntimeError(
                f"Existing {destination} failed SHA-256 validation; rerun with --force."
            )
        print(f"Using existing {destination} (sha256={digest})")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(asset.url, headers={"User-Agent": "drone-atlas/0.1"})
    print(f"Downloading {asset.url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {asset.filename}: {exc}") from exc

    digest = sha256_file(temporary)
    if asset.sha256 and digest != asset.sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {asset.filename}: expected {asset.sha256}, got {digest}"
        )
    temporary.replace(destination)
    validation = "verified" if asset.sha256 else "recorded (upstream publishes no stable SHA-256)"
    print(f"Saved {destination} (sha256={digest}, {validation})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("yolov3",), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--force", action="store_true", help="replace existing assets")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for asset in YOLOV3_ASSETS:
            download(asset, args.output_dir / asset.filename, force=args.force)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("Model assets are ready. The models directory is excluded by .gitignore.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
