from pathlib import Path

import pytest

from drone_atlas import cli
from drone_atlas.media import MediaResult


def test_cli_rejects_invalid_threshold():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            ["image", "--input", "in.jpg", "--output", "out.jpg", "--confidence", "1.1"]
        )

    assert exc_info.value.code == 2


def test_cli_rejects_missing_input(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "image",
                "--input",
                str(tmp_path / "missing.jpg"),
                "--output",
                str(tmp_path / "out.jpg"),
            ]
        )

    assert exc_info.value.code == 2


def test_image_command_wires_paths_and_thresholds(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"test")
    output_path = tmp_path / "output.jpg"
    captured = {}

    class FakeDetector:
        def __init__(self, paths, **kwargs):
            captured["paths"] = paths
            captured["kwargs"] = kwargs

    def fake_runner(source: Path, destination: Path, detector: FakeDetector):
        captured["source"] = source
        captured["destination"] = destination
        return MediaResult(1, 3, destination)

    monkeypatch.setattr(cli, "YoloV3Detector", FakeDetector)
    monkeypatch.setattr(cli, "process_image", fake_runner)

    result = cli.main(
        [
            "image",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model-dir",
            str(tmp_path / "model-files"),
            "--confidence",
            "0.6",
            "--nms",
            "0.3",
        ]
    )

    assert result == 0
    assert captured["paths"].weights == tmp_path / "model-files" / "yolov3.weights"
    assert captured["kwargs"] == {"confidence_threshold": 0.6, "nms_threshold": 0.3}
    assert captured["source"] == input_path
    assert "found 3 detection(s)" in capsys.readouterr().out
