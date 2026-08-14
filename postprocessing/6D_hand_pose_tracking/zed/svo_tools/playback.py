########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""Play back an SVO recording, optionally writing the left view to a video file.

Adapted from the Stereolabs SVO playback sample. Press ``s`` while playing to
save the current frame as an image, and ``q`` to quit.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import pyzed.sl as sl

from HandTrackingModule.Zed import (
    DEPTH_MODE_CHOICES,
    get_camera_fps,
    get_resolution_wh,
    is_end_of_svo,
    resolve_depth_mode,
    set_fill_sensing_mode,
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="playback.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python playback.py take1.svo2
  python playback.py take1.svo2 --output take1.mp4
  python playback.py take1.svo2 --no-display --output take1.mp4 --info
""",
    )
    parser.add_argument("input", help="Path to the SVO file to play back.")
    parser.add_argument(
        "-o", "--output",
        help="Write the left view to this video file. Default: no video is written.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec used for --output (default: %(default)s).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Frame rate of the output video. Default: the recording's frame rate.",
    )
    parser.add_argument(
        "--depth-mode",
        default="QUALITY",
        choices=DEPTH_MODE_CHOICES,
        help="Depth mode used during playback (default: %(default)s).",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=".",
        help="Directory the 's' key saves snapshots into (default: current directory).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window; useful for headless conversion.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print detailed camera information when playback finishes.",
    )
    return parser


def print_camera_information(cam):
    print()
    try:
        info = cam.get_camera_information()
        calibration = getattr(info, "calibration_parameters", None)
        if calibration is None:
            configuration = getattr(info, "camera_configuration", None)
            calibration = getattr(configuration, "calibration_parameters", None)
        if calibration is not None:
            print("Distortion factor of the right cam after calibration: {0}.".format(
                calibration.right_cam.disto))

        init_params = cam.get_init_parameters()
        print("Depth min and max range values: {0}, {1}".format(
            init_params.depth_minimum_distance, init_params.depth_maximum_distance))

        width, height = get_resolution_wh(info)
        print("Resolution: {0}, {1}.".format(width, height))
        print("Camera FPS: {0}".format(get_camera_fps(info)))
        print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
    except Exception as exc:  # noqa: BLE001 - informational output only
        print("Could not read all camera information: {0}".format(exc))


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.input):
        print("Error: input file not found: {0}".format(args.input), file=sys.stderr)
        return 2

    print("Reading SVO file: {0}".format(args.input))

    input_type = sl.InputType()
    input_type.set_from_svo_file(args.input)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    try:
        init.depth_mode, _ = resolve_depth_mode(args.depth_mode)
    except ValueError as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        return 2

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Unable to open the SVO file: {0}".format(repr(status)), file=sys.stderr)
        return 1

    runtime = sl.RuntimeParameters()
    set_fill_sensing_mode(runtime)
    mat = sl.Mat()

    camera_info = cam.get_camera_information()
    width, height = get_resolution_wh(camera_info)
    total_frames = cam.get_svo_number_of_frames()

    writer = None
    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        if directory:
            os.makedirs(directory, exist_ok=True)
        fps = args.fps or get_camera_fps(camera_info)
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*args.codec), fps, (width, height)
        )
        if not writer.isOpened():
            print(
                "Error: could not open the video writer. Check the output path and "
                "that the '{0}' codec is available.".format(args.codec),
                file=sys.stderr,
            )
            cam.close()
            return 1

    if not args.no_display:
        print("  Save the current image:     s")
        print("  Quit the video reading:     q\n")

    os.makedirs(args.snapshot_dir, exist_ok=True)
    frame = 0
    try:
        while True:
            err = cam.grab(runtime)
            if err != sl.ERROR_CODE.SUCCESS:
                if is_end_of_svo(err):
                    print("\nEnd of SVO file reached.")
                else:
                    print("\nZED error: {0}".format(err))
                break

            frame += 1
            print("Current frame: {0}/{1}".format(frame, total_frames), end="\r")

            cam.retrieve_image(mat, sl.VIEW.LEFT)
            img = mat.get_data()
            # The ZED returns BGRA; the video writer expects 3 channels.
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img

            if writer is not None:
                writer.write(img_bgr)

            if not args.no_display:
                preview = img_bgr.copy()
                cv2.putText(
                    preview, str(frame), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3,
                )
                cv2.imshow("ZED", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    path = os.path.join(args.snapshot_dir, "frame_{0:06d}.png".format(frame))
                    cv2.imwrite(path, img_bgr)
                    print("\nSaved snapshot: {0}".format(path))

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            print("\nVideo written to: {0}".format(args.output))
        if args.info:
            print_camera_information(cam)
        cam.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
