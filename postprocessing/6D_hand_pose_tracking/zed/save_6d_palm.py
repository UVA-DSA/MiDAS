"""Extract 6D palm poses and fingertip positions from an SVO file or a live ZED camera.

For every processed frame the script writes one CSV row containing, per hand:
the palm centroid (x, y, z), the palm orientation (yaw, pitch, roll in
degrees), and the index and thumb fingertip positions. Coordinates are
expressed in the ZED camera frame, in the unit selected with ``--units``
(metres by default).

Run ``python save_6d_palm.py --help`` for the full list of options.
"""

import argparse
import os
import sys

# Allow running this file directly (python zed/save_6d_palm.py) as well as
# from another working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import pandas as pd
import pyzed.sl as sl

from HandTrackingModule.HandTracking import INDEX_TIP, NUM_LANDMARKS, THUMB_TIP
from HandTrackingModule.Zed import is_end_of_svo
from HandTrackingModule.cli import (
    SVO_EXTENSIONS,
    add_camera_arguments,
    add_tracking_arguments,
    build_tracker,
    open_camera,
    positive_int,
)

HANDS = ("left", "right")


def _column_names(with_landmarks=False):
    """Column order of the produced CSV."""
    columns = ["frame", "timestamp_ns"]
    for hand in HANDS:
        columns.append("{0}_detected".format(hand))
        columns += ["{0}_palm_{1}".format(hand, axis) for axis in "xyz"]
        columns += [
            "{0}_yaw_deg".format(hand),
            "{0}_pitch_deg".format(hand),
            "{0}_roll_deg".format(hand),
        ]
        columns += ["{0}_index_tip_{1}".format(hand, axis) for axis in "xyz"]
        columns += ["{0}_thumb_tip_{1}".format(hand, axis) for axis in "xyz"]
    if with_landmarks:
        for hand in HANDS:
            for landmark in range(NUM_LANDMARKS):
                columns += [
                    "{0}_lm{1:02d}_{2}".format(hand, landmark, axis) for axis in "xyz"
                ]
    return columns


def _hand_row(prefix, data, detector, with_landmarks):
    """Build the CSV fields for a single hand.

    `data` is the ``(21, 3)`` landmark array, or an empty array when the hand
    was not detected. Undetected hands are reported as zeros with
    ``<hand>_detected = 0``.
    """
    detected = data.shape == (NUM_LANDMARKS, 3)
    row = {"{0}_detected".format(prefix): int(detected)}

    if detected:
        centroid = detector.calculate_centroid(data)
        orientation = detector.calculate_orientation(data)
        index_tip = data[INDEX_TIP]
        thumb_tip = data[THUMB_TIP]
    else:
        centroid = (0.0, 0.0, 0.0)
        orientation = (0.0, 0.0, 0.0)
        index_tip = (0.0, 0.0, 0.0)
        thumb_tip = (0.0, 0.0, 0.0)

    for axis_index, axis in enumerate("xyz"):
        row["{0}_palm_{1}".format(prefix, axis)] = float(centroid[axis_index])
        row["{0}_index_tip_{1}".format(prefix, axis)] = float(index_tip[axis_index])
        row["{0}_thumb_tip_{1}".format(prefix, axis)] = float(thumb_tip[axis_index])

    for angle_index, angle in enumerate(("yaw", "pitch", "roll")):
        row["{0}_{1}_deg".format(prefix, angle)] = float(orientation[angle_index])

    if with_landmarks:
        for landmark in range(NUM_LANDMARKS):
            for axis_index, axis in enumerate("xyz"):
                value = float(data[landmark][axis_index]) if detected else 0.0
                row["{0}_lm{1:02d}_{2}".format(prefix, landmark, axis)] = value

    return row


def process(source, args):
    """Track hands through `source` and return the results as a DataFrame.

    Args:
        source: Path to an SVO recording, or ``None`` for the live camera.
        args: Parsed command line arguments (see :func:`build_parser`).
    """
    detector = build_tracker(args)
    cam = open_camera(args, filename=source, verbose=not args.quiet)

    try:
        if not args.quiet:
            cam.print_information()

        total_frames = cam.get_number_of_frames() if cam.svo_mode else -1
        if cam.svo_mode and args.start_frame:
            cam.set_position(args.start_frame)

        rows = []
        grabbed = 0
        interrupted = False

        try:
            while True:
                err = cam.grab()
                if err != sl.ERROR_CODE.SUCCESS:
                    if is_end_of_svo(err):
                        if not args.quiet:
                            print("\nEnd of SVO file reached.")
                    else:
                        print("\nZED error after {0} frames: {1}".format(grabbed, err))
                    break

                position = cam.get_position() if cam.svo_mode else grabbed
                grabbed += 1

                if args.end_frame is not None and cam.svo_mode and position > args.end_frame:
                    break

                # Process every Nth frame only.
                if (grabbed - 1) % args.frame_skip != 0:
                    continue

                cam.get_image()
                img = cam.img
                depth_img = cam.depth_img
                pcl = cam.point_cloud

                annotate = args.display and not args.no_annotate
                img = detector.findHands(img, draw=annotate)
                data_left, data_right = detector.find_positions(
                    depth_img,
                    pcl,
                    cam.camera_params,
                    draw=annotate,
                    verbose=not args.quiet,
                )

                row = {"frame": position, "timestamp_ns": cam.get_timestamp_ns()}
                row.update(_hand_row("left", data_left, detector, args.landmarks))
                row.update(_hand_row("right", data_right, detector, args.landmarks))
                rows.append(row)

                if args.display:
                    cv2.imshow("Hand tracking", img)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        interrupted = True
                        break

                if not args.quiet:
                    total = total_frames if total_frames > 0 else "?"
                    print(" | Frame {0}/{1}".format(position, total), end="\r")

                if args.max_frames is not None and len(rows) >= args.max_frames:
                    break

        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted, saving what was processed so far...")

        if interrupted and not args.quiet:
            print("\nStopped after {0} processed frames.".format(len(rows)))

        return pd.DataFrame(rows, columns=_column_names(args.landmarks))

    finally:
        detector.close()
        cam.close()
        if args.display:
            cv2.destroyAllWindows()


def default_output_path(source, output_dir=None):
    """CSV path used when ``--output`` is not given."""
    if source is None:
        name = "hand_pose_live.csv"
        directory = output_dir or os.getcwd()
    else:
        name = os.path.splitext(os.path.basename(source))[0] + ".csv"
        directory = output_dir or os.path.dirname(os.path.abspath(source))
    return os.path.join(directory, name)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="save_6d_palm.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # process one recording, CSV written next to it
  python save_6d_palm.py /data/session01/take1.svo2

  # choose the output file and process every 2nd frame
  python save_6d_palm.py take1.svo2 -o results/take1.csv --frame-skip 2

  # live camera, watch the tracking, stop with q or Ctrl+C
  python save_6d_palm.py --live --display -o live.csv

  # also record all 21 landmarks per hand
  python save_6d_palm.py take1.svo2 --landmarks
""",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to an SVO recording ({0}). Omit it, or pass --live, to use "
             "the connected ZED camera.".format(" / ".join(SVO_EXTENSIONS)),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Capture from the connected ZED camera instead of a file.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV path. Default: <input name>.csv next to the input "
             "file, or hand_pose_live.csv in the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for the default output name. Ignored when --output is given.",
    )

    processing = parser.add_argument_group("processing options")
    processing.add_argument(
        "--frame-skip",
        type=positive_int,
        default=1,
        help="Process every Nth grabbed frame; 1 processes every frame "
             "(default: %(default)s).",
    )
    processing.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First SVO frame to process (default: %(default)s).",
    )
    processing.add_argument(
        "--end-frame",
        type=int,
        help="Last SVO frame to process. Default: run to the end of the file.",
    )
    processing.add_argument(
        "--max-frames",
        type=positive_int,
        help="Stop after this many processed frames. Useful for a quick test "
             "run, and to bound live captures.",
    )
    processing.add_argument(
        "--landmarks",
        action="store_true",
        help="Also write all 21 landmarks per hand (adds 126 columns).",
    )

    add_camera_arguments(parser)
    add_tracking_arguments(parser)

    output = parser.add_argument_group("display / logging options")
    output.add_argument(
        "--display",
        action="store_true",
        help="Show the annotated video while processing. Slower; press q to stop.",
    )
    output.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip drawing the landmark overlay even when --display is used.",
    )
    output.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    source = None if args.live else args.input
    if source is not None:
        if not os.path.isfile(source):
            print("Error: input file not found: {0}".format(source), file=sys.stderr)
            return 2
        print("Camera mode: SVO")
    else:
        if args.input:
            print(
                "Error: --live was given together with an input file.",
                file=sys.stderr,
            )
            return 2
        print("Camera mode: live streaming")

    try:
        df = process(source, args)
    except (RuntimeError, ValueError) as exc:
        print("\nError: {0}".format(exc), file=sys.stderr)
        return 1

    output_path = args.output or default_output_path(source, args.output_dir)
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("\nSaved {0} frames to: {1}".format(len(df), output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
