# measurement_engine

A Python-based measurement engine that lets tailors extract body measurements
from photographs using computer vision.

## How it works

1. A full-body, front-view photograph of the customer is passed to the engine.
2. [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
   detects 33 body landmarks in the image.
3. Key distances (shoulder width, torso length, inseam, sleeve length, etc.)
   are calculated from those landmarks in pixels.
4. If the person's height is known, a pixel-to-centimetre scale factor is
   derived automatically so that all measurements are returned in **cm**.
5. Circumference measurements (chest, waist, hip, neck) are estimated using
   an ellipse model that combines the visible width with typical depth-to-width
   body proportions.

### Measurements provided

| Measurement           | Description                                          |
|-----------------------|------------------------------------------------------|
| `shoulder_width`      | Distance between left and right shoulder joints     |
| `chest_circumference` | Estimated chest/bust girth                          |
| `waist_circumference` | Estimated waist girth                               |
| `hip_circumference`   | Estimated hip girth                                 |
| `inseam_length`       | Hip midpoint to ankle midpoint                      |
| `sleeve_length`       | Shoulder → elbow → wrist path length                |
| `torso_length`        | Shoulder midpoint to hip midpoint                   |
| `back_length`         | Same as torso length (for a standing person)        |
| `total_height`        | Estimated full body height                          |
| `neck_circumference`  | Estimated neck girth                                |

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

### Python API

```python
from measurement_engine import MeasurementEngine

engine = MeasurementEngine()

# Measurements in pixels (no height reference)
result = engine.analyze("front_view.jpg")
print(result)

# Measurements in centimetres (height calibration)
result = engine.analyze("front_view.jpg", person_height_cm=175.0)
print(result)

# With annotated visualisation saved to disk
result, annotated_image = engine.analyze_with_visualization(
    "front_view.jpg",
    person_height_cm=175.0,
    output_path="annotated.jpg",
)
```

### Command-line interface

```bash
# Print measurements in pixels
measurement-engine analyze front_view.jpg

# Print measurements in centimetres
measurement-engine analyze front_view.jpg --height 175

# Save an annotated visualisation
measurement-engine analyze front_view.jpg --height 175 --output annotated.jpg
```

## Tips for best results

* Use a **full-body, front-view** photograph.
* The person should stand **upright** with arms slightly away from the body.
* Ensure **good, even lighting** with a plain or contrasting background.
* The more of the body that is visible, the more measurements can be computed.

## Running the tests

```bash
pip install pytest
pytest
```
