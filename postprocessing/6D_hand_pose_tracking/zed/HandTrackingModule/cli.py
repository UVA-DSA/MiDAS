"""Shared command line options for the scripts in this package.

Every executable script exposes the same camera and hand-tracking switches, so
they are declared once here and reused. Nothing in this module is specific to
a machine, a dataset layout or a file naming scheme.
"""

import argparse

from .HandTracking import HandTracking
from .Zed import (
    COORDINATE_UNIT_CHOICES,
    DEFAULT_COORDINATE_UNITS,
    DEFAULT_DEPTH_CONFIDENCE,
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_MODE,
    DEFAULT_FPS,
    DEFAULT_RESOLUTION,
    DEPTH_MODE_CHOICES,
    RESOLUTION_CHOICES,
    Zed,
)

#: Recording extensions understood by the ZED SDK.
SVO_EXTENSIONS = (".svo2", ".svo")


def add_camera_arguments(parser):
    """Add the ZED capture options to `parser`."""
    group = parser.add_argument_group("camera / SVO options")
    group.add_argument(
        "--resolution",
        default=DEFAULT_RESOLUTION,
        choices=RESOLUTION_CHOICES,
        help="Capture resolution for the live camera; ignored when replaying "
             "an SVO (default: %(default)s).",
    )
    group.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Capture frame rate for the live camera (default: %(default)s).",
    )
    group.add_argument(
        "--depth-mode",
        default=DEFAULT_DEPTH_MODE,
        choices=DEPTH_MODE_CHOICES,
        help="ZED depth mode. Neural modes need the AI models installed, see "
             "download_zed_models.py (default: %(default)s).",
    )
    group.add_argument(
        "--units",
        default=DEFAULT_COORDINATE_UNITS,
        choices=COORDINATE_UNIT_CHOICES,
        help="Unit of the 3D coordinates written to the output (default: %(default)s).",
    )
    group.add_argument(
        "--depth-min",
        type=float,
        default=DEFAULT_DEPTH_MIN,
        help="Minimum depth distance, in --units (default: %(default)s).",
    )
    group.add_argument(
        "--depth-max",
        type=float,
        default=DEFAULT_DEPTH_MAX,
        help="Maximum depth distance, in --units (default: %(default)s).",
    )
    group.add_argument(
        "--depth-confidence",
        type=int,
        default=DEFAULT_DEPTH_CONFIDENCE,
        help="Depth confidence threshold, 0-100. Lower discards more uncertain "
             "depth pixels (default: %(default)s).",
    )
    group.add_argument(
        "--svo-real-time",
        action="store_true",
        help="Replay the SVO at its recorded frame rate instead of as fast as "
             "possible. Drops frames, so leave this off for batch processing.",
    )
    return group


def add_tracking_arguments(parser):
    """Add the MediaPipe hand tracking options to `parser`."""
    group = parser.add_argument_group("hand tracking options")
    group.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands tracked per frame (default: %(default)s).",
    )
    group.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence, 0-1. Higher is stricter "
             "and faster (default: %(default)s).",
    )
    group.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.7,
        help="MediaPipe minimum tracking confidence, 0-1 (default: %(default)s).",
    )
    group.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1),
        default=1,
        help="MediaPipe hand model complexity: 0 is faster, 1 is more accurate "
             "(default: %(default)s).",
    )
    group.add_argument(
        "--swap-handedness",
        action="store_true",
        help="Swap the reported left and right hands. Use this if your "
             "recording is mirrored and the hands come out the wrong way round.",
    )
    return group


def open_camera(args, filename=None, verbose=True):
    """Build a :class:`~HandTrackingModule.Zed.Zed` from parsed `args`."""
    return Zed(
        filename=filename,
        depth_confidence=args.depth_confidence,
        resolution=args.resolution,
        fps=args.fps,
        depth_mode=args.depth_mode,
        coordinate_units=args.units,
        depth_minimum_distance=args.depth_min,
        depth_maximum_distance=args.depth_max,
        svo_real_time_mode=getattr(args, "svo_real_time", False),
        verbose=verbose,
    )


def build_tracker(args):
    """Build a :class:`~HandTrackingModule.HandTracking.HandTracking` from `args`."""
    return HandTracking(
        maxHands=args.max_hands,
        detectionCon=args.detection_confidence,
        trackCon=args.tracking_confidence,
        modelComplexity=args.model_complexity,
        swapHandedness=args.swap_handedness,
    )


def positive_int(value):
    """argparse type for a strictly positive integer."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer, got {0}".format(value))
    return number
