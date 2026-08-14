"""Live preview of the tracked hands: video, depth view and a matplotlib 3D skeleton.

Use this to check that the camera, the depth range and the tracking thresholds
are set up correctly before running a batch. Optionally saves every displayed
frame and plot to disk.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyzed.sl as sl

from HandTrackingModule.HandTracking import NUM_LANDMARKS
from HandTrackingModule.Zed import is_end_of_svo
from HandTrackingModule.cli import (
    SVO_EXTENSIONS,
    add_camera_arguments,
    add_tracking_arguments,
    build_tracker,
    open_camera,
)


def _autoscale_axes(ax, data_plot, zoom):
    """Centre the 3D axes on the detected landmarks."""
    if data_plot.size == 0:
        return
    finite_points = data_plot[np.isfinite(data_plot).all(axis=1)]
    if finite_points.size == 0:
        return
    mins = np.min(finite_points, axis=0)
    maxs = np.max(finite_points, axis=0)
    if not (np.isfinite(mins).all() and np.isfinite(maxs).all()):
        return
    center = (mins + maxs) / 2.0
    half_range = float(np.max((maxs - mins) / 2.0))
    if not np.isfinite(half_range) or half_range <= 0:
        half_range = 1e-6
    zoomed = half_range / zoom
    if np.isfinite(zoomed) and zoomed > 0 and np.isfinite(center).all():
        ax.set_xlim(center[0] - zoomed, center[0] + zoomed)
        ax.set_ylim(center[1] - zoomed, center[1] + zoomed)
        ax.set_zlim(center[2] - zoomed, center[2] + zoomed)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="show3d_matplot.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python show3d_matplot.py                       # live camera
  python show3d_matplot.py take1.svo2            # replay a recording
  python show3d_matplot.py take1.svo2 --frame 250    # inspect a single frame
  python show3d_matplot.py take1.svo2 --save-dir out/take1
""",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to an SVO recording ({0}). Omit to use the live camera.".format(
            " / ".join(SVO_EXTENSIONS)
        ),
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Show a single SVO frame by index and wait for a key press.",
    )
    parser.add_argument(
        "--save-dir",
        help="Write images, depth views and plots under this directory.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor applied to the displayed windows (default: %(default)s).",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.5,
        help="Zoom factor of the 3D plot around the detected hands (default: %(default)s).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the matplotlib 3D view (faster).",
    )
    add_camera_arguments(parser)
    add_tracking_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.input is not None and not os.path.isfile(args.input):
        print("Error: input file not found: {0}".format(args.input), file=sys.stderr)
        return 2
    if args.frame is not None and args.input is None:
        print(
            "Error: --frame is only supported when reading from an SVO file.",
            file=sys.stderr,
        )
        return 2

    print("Camera mode: {0}".format("SVO" if args.input else "live streaming"))

    detector = build_tracker(args)
    try:
        cam = open_camera(args, filename=args.input)
    except (RuntimeError, ValueError) as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        detector.close()
        return 1

    try:
        cam.print_information()

        images_dir = depth_dir = plots_dir = None
        if args.save_dir:
            images_dir = os.path.join(args.save_dir, "images")
            depth_dir = os.path.join(args.save_dir, "depth")
            plots_dir = os.path.join(args.save_dir, "plots")
            for directory in (images_dir, depth_dir, plots_dir):
                os.makedirs(directory, exist_ok=True)

        if args.frame is not None:
            total_frames = cam.get_number_of_frames()
            index = args.frame
            if total_frames and total_frames > 0:
                index = max(0, min(index, total_frames - 1))
            print("Seeking to frame {0}/{1}".format(index, total_frames - 1 if total_frames else "?"))
            cam.set_position(index)

        show_plot = not args.no_plot
        ax = None
        if show_plot:
            fig = plt.figure()
            plt.ion()
            ax = fig.add_subplot(111, projection="3d")

        frame_counter = 0
        while True:
            err = cam.grab()
            if err != sl.ERROR_CODE.SUCCESS:
                if is_end_of_svo(err):
                    print("\nEnd of SVO file reached.")
                else:
                    print("\nZED error: {0}".format(err))
                break

            cam.get_image()
            img = cam.img
            depth_img = cam.depth_img
            pcl = cam.point_cloud

            img = detector.findHands(img)
            data_left, data_right = detector.find_positions(depth_img, pcl, cam.camera_params)

            if args.scale != 1.0:
                img = cv2.resize(img, (0, 0), None, args.scale, args.scale)
                depth_img = cv2.resize(depth_img, (0, 0), None, args.scale, args.scale)

            if show_plot:
                full = (NUM_LANDMARKS, 3)
                if data_right.shape == full and data_left.shape == full:
                    data_plot = np.vstack((data_right, data_left))
                elif data_right.shape == full:
                    data_plot = data_right
                elif data_left.shape == full:
                    data_plot = data_left
                else:
                    data_plot = np.array([])
                detector.plot(ax, plt, data_plot)
                _autoscale_axes(ax, data_plot, args.zoom)

            detector.displayFPS(img)
            cv2.imshow("Image", img)
            cv2.imshow("Depth", depth_img)

            if args.save_dir:
                name = "{0:06d}".format(frame_counter)
                cv2.imwrite(os.path.join(images_dir, "frame_{0}.png".format(name)), img)
                cv2.imwrite(os.path.join(depth_dir, "depth_{0}.png".format(name)), depth_img)
                if show_plot:
                    plt.savefig(os.path.join(plots_dir, "plot_{0}.png".format(name)))
            frame_counter += 1

            if args.frame is not None:
                # Show only the requested frame and wait for a key before exiting.
                cv2.waitKey(0)
                break
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    finally:
        detector.close()
        cam.close()
        cv2.destroyAllWindows()
        plt.close("all")


if __name__ == "__main__":
    sys.exit(main())
