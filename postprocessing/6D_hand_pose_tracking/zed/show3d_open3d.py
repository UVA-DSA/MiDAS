"""Live preview of the tracked hands with an Open3D point cloud viewer.

Same purpose as ``show3d_matplot.py`` but renders the 3D hand landmarks in an
Open3D window, which is faster and easier to navigate than the matplotlib
view. The left hand is drawn in blue and the right hand in red.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import pyzed.sl as sl

from HandTrackingModule.Vis3D import Vis3D
from HandTrackingModule.Zed import is_end_of_svo
from HandTrackingModule.cli import (
    SVO_EXTENSIONS,
    add_camera_arguments,
    add_tracking_arguments,
    build_tracker,
    open_camera,
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="show3d_open3d.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python show3d_open3d.py                # live camera
  python show3d_open3d.py take1.svo2     # replay a recording
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
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor applied to the displayed windows (default: %(default)s).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only show the Open3D window, without the image and depth windows.",
    )
    add_camera_arguments(parser)
    add_tracking_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.input is not None and not os.path.isfile(args.input):
        print("Error: input file not found: {0}".format(args.input), file=sys.stderr)
        return 2

    print("Camera mode: {0}".format("SVO" if args.input else "live streaming"))

    detector = build_tracker(args)
    try:
        cam = open_camera(args, filename=args.input)
    except (RuntimeError, ValueError) as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        detector.close()
        return 1

    vis3d = Vis3D()
    try:
        cam.print_information()

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

            vis3d.show_hand(data_left, vis3d.blue)
            vis3d.show_hand(data_right, vis3d.red)

            if not args.no_video:
                if args.scale != 1.0:
                    img = cv2.resize(img, (0, 0), None, args.scale, args.scale)
                    depth_img = cv2.resize(depth_img, (0, 0), None, args.scale, args.scale)

                detector.displayFPS(img)
                cv2.imshow("Image", img)
                cv2.imshow("Depth", depth_img)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    finally:
        detector.close()
        cam.close()
        vis3d.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
