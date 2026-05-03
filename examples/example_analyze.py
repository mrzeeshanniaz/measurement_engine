"""
Example: Analyse a photograph and print body measurements.

Usage
-----
    python example_analyze.py <path_to_image> [--height <cm>]

The image should be a full-body, front-view photograph of a person
standing upright in good lighting.
"""

import argparse
import sys
from pathlib import Path

# Allow running from the examples/ directory without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from measurement_engine import MeasurementEngine


def main():
    parser = argparse.ArgumentParser(description="Body measurement from image")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        metavar="CM",
        help="Known height of the person in centimetres",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Path to save an annotated visualisation image",
    )
    args = parser.parse_args()

    engine = MeasurementEngine()

    try:
        if args.output:
            result, _ = engine.analyze_with_visualization(
                args.image,
                person_height_cm=args.height,
                output_path=args.output,
            )
            print(f"Annotated image saved to: {args.output}")
        else:
            result = engine.analyze(args.image, person_height_cm=args.height)

        print(result)

    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
