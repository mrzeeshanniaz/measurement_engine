"""
Command-line interface for the measurement engine.

Usage examples::

    # Measurements in pixels (no height reference)
    measurement-engine analyze photo.jpg

    # Measurements in centimetres (height calibration)
    measurement-engine analyze photo.jpg --height 175

    # Save annotated visualisation
    measurement-engine analyze photo.jpg --height 175 --output annotated.jpg
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measurement-engine",
        description="Extract body measurements from a photograph for tailoring.",
    )
    sub = parser.add_subparsers(dest="command")

    analyze = sub.add_parser(
        "analyze",
        help="Analyse a photograph and print body measurements.",
    )
    analyze.add_argument(
        "image",
        help="Path to the input image (front-view, full-body photograph).",
    )
    analyze.add_argument(
        "--height",
        type=float,
        default=None,
        metavar="CM",
        help="Known height of the person in centimetres (enables cm output).",
    )
    analyze.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Path to save an annotated visualisation image.",
    )

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "analyze":
        from measurement_engine import MeasurementEngine  # noqa: PLC0415

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
            return 0

        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
