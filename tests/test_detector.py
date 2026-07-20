import pytest

from drone_atlas.detector import (
    class_color,
    decode_yolo_outputs,
    normalize_nms_indices,
)


class ArrayLike:
    def tolist(self):
        return [[2], [0], [2]]


def test_decodes_box_and_combines_objectness_with_class_score():
    outputs = [[[0.5, 0.5, 0.4, 0.2, 0.8, 0.1, 0.9]]]

    detections = decode_yolo_outputs(outputs, (100, 200, 3), 0.5)

    assert len(detections) == 1
    assert detections[0].class_id == 1
    assert detections[0].confidence == pytest.approx(0.72)
    assert detections[0].box == (60, 40, 80, 20)


def test_rejects_low_combined_confidence():
    outputs = [[[0.5, 0.5, 0.2, 0.2, 0.4, 0.99]]]

    assert decode_yolo_outputs(outputs, (100, 100, 3), 0.5) == []


def test_clips_boxes_to_image_bounds():
    outputs = [[[0.0, 0.0, 0.4, 0.4, 1.0, 1.0]]]

    detection = decode_yolo_outputs(outputs, (100, 100, 3), 0.5)[0]

    assert detection.box == (0, 0, 20, 20)


def test_normalizes_nms_shapes_and_removes_duplicates():
    assert normalize_nms_indices(None) == []
    assert normalize_nms_indices(3) == [3]
    assert normalize_nms_indices(((1,), [4])) == [1, 4]
    assert normalize_nms_indices(ArrayLike()) == [2, 0]


def test_class_colours_are_deterministic_and_visible():
    assert class_color(7) == class_color(7)
    assert class_color(7) != class_color(8)
    assert all(64 <= component <= 255 for component in class_color(7))
