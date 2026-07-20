# Demo asset provenance

## `input.jpg`

- Title: **Car on street**
- Creator/source: U.S. Fish and Wildlife Service
- Licence: public domain in the United States (U.S. federal government work)
- Source page: <https://commons.wikimedia.org/wiki/File:Car_on_street.jpg>
- Retrieved: 2026-07-20, using Wikimedia Commons' 1280-pixel redirect
- SHA-256: `63e3568e7057314abcc9956eef007692620a59d5e73162a23948c30b1c9f1592`
- Modification: Wikimedia-generated resize only

## `detected.jpg`

Generated locally from `input.jpg` with the official pretrained YOLOv3 weights and:

```bash
python -m drone_atlas image \
  --input assets/demo/input.jpg \
  --output assets/demo/detected.jpg \
  --model-dir models
```

The output contains two detections at the default thresholds. It is included solely to demonstrate this repository's inference and annotation path; it is not an accuracy benchmark.

The original personal images and videos used during early experimentation are intentionally excluded.
