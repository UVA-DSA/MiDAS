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

"""Export an SVO recording to a side-by-side video or to an image sequence.

Adapted from the Stereolabs SVO export sample. The export mode is selected by
name instead of by a magic number.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import numpy as np
import pyzed.sl as sl

from HandTrackingModule.Zed import (
    COORDINATE_UNIT_CHOICES,
    get_camera_fps,
    get_resolution_wh,
    is_end_of_svo,
    resolve_enum,
    set_fill_sensing_mode,
)

#: Export mode -> (writes a video, right-hand pane content)
MODES = {
    "video-left-right": (True, "RIGHT"),
    "video-left-depth": (True, "DEPTH"),
    "images-left-right": (False, "RIGHT"),
    "images-left-depth": (False, "DEPTH"),
    "images-left-depth16": (False, "DEPTH16"),
}

#: Numeric aliases kept for compatibility with the original Stereolabs sample.
LEGACY_MODES = {
    "0": "video-left-right",
    "1": "video-left-depth",
    "2": "images-left-right",
    "3": "images-left-depth",
    "4": "images-left-depth16",
}


def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %.1f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="save_to_avi.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""export modes:
  video-left-right     side-by-side left + right video       (alias 0)
  video-left-depth     side-by-side left + depth view video  (alias 1)
  images-left-right    left/right PNG sequence               (alias 2)
  images-left-depth    left/depth-view PNG sequence          (alias 3)
  images-left-depth16  left PNG + 16-bit depth PNG sequence  (alias 4)

examples:
  python save_to_avi.py take1.svo2 take1.avi --mode video-left-right
  python save_to_avi.py take1.svo2 frames/ --mode images-left-depth16
""",
    )
    parser.add_argument("input", help="Path to the SVO file to export.")
    parser.add_argument(
        "output",
        help="Output video file, or output directory for the image sequence modes.",
    )
    parser.add_argument(
        "--mode",
        default="video-left-right",
        choices=sorted(MODES) + sorted(LEGACY_MODES),
        help="What to export (default: %(default)s).",
    )
    parser.add_argument(
        "--codec",
        default="M4S2",
        help="FourCC codec used for the video modes (default: %(default)s).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Frame rate of the output video. Default: the recording's frame rate.",
    )
    parser.add_argument(
        "--units",
        default="MILLIMETER",
        choices=COORDINATE_UNIT_CHOICES,
        help="Unit of the 16-bit depth output (default: %(default)s).",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    mode = LEGACY_MODES.get(args.mode, args.mode)
    output_as_video, right_pane = MODES[mode]

    svo_input_path = Path(args.input)
    output_path = Path(args.output)

    if not svo_input_path.is_file():
        print("Error: input file not found: {0}".format(svo_input_path), file=sys.stderr)
        return 2

    if output_as_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    try:
        init_params.coordinate_units, _ = resolve_enum(sl.UNIT, (args.units,), "coordinate unit")
    except ValueError as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        return 2

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Unable to open the SVO file: {0}".format(repr(err)), file=sys.stderr)
        zed.close()
        return 1

    # Get image size
    camera_info = zed.get_camera_information()
    width, height = get_resolution_wh(camera_info)
    width_sbs = width * 2

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()

    video_writer = None
    if output_as_video:
        fps = args.fps or max(get_camera_fps(camera_info), 25)
        video_writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*args.codec), fps, (width_sbs, height)
        )

        if not video_writer.isOpened():
            print(
                "Error: OpenCV video writer cannot be opened. Check the output path, "
                "write permissions, and that the '{0}' codec is available.".format(args.codec),
                file=sys.stderr,
            )
            zed.close()
            return 1

    rt_param = sl.RuntimeParameters()
    set_fill_sensing_mode(rt_param)

    # Start SVO conversion to video / image sequence
    print("Converting SVO... Use Ctrl-C to interrupt conversion.")

    nb_frames = zed.get_svo_number_of_frames()

    try:
        while True:
            err = zed.grab(rt_param)
            if err != sl.ERROR_CODE.SUCCESS:
                if is_end_of_svo(err):
                    print("\nSVO end has been reached. Exiting now.")
                else:
                    print("\nZED error: {0}".format(err))
                break

            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            if right_pane == "RIGHT":
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            elif right_pane == "DEPTH":
                zed.retrieve_image(right_image, sl.VIEW.DEPTH)
            else:  # DEPTH16
                zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            if output_as_video:
                # Copy the left image to the left side of SBS image
                svo_image_sbs_rgba[0:height, 0:width, :] = left_image.get_data()

                # Copy the right image to the right side of SBS image
                svo_image_sbs_rgba[0:, width:, :] = right_image.get_data()

                # Convert SVO image from RGBA to RGB
                ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba, cv2.COLOR_RGBA2RGB)

                # Write the RGB image in the video
                video_writer.write(ocv_image_sbs_rgb)
            else:
                # Generate file names
                index = str(svo_position).zfill(6)
                filename1 = output_path / "left{0}.png".format(index)
                filename2 = output_path / "{0}{1}.png".format(
                    "right" if right_pane == "RIGHT" else "depth", index
                )

                # Save Left images
                cv2.imwrite(str(filename1), left_image.get_data())

                if right_pane != "DEPTH16":
                    # Save right images
                    cv2.imwrite(str(filename2), right_image.get_data())
                else:
                    # Save depth images (convert to uint16)
                    cv2.imwrite(str(filename2), depth_image.get_data().astype(np.uint16))

            # Display progress
            if nb_frames > 0:
                progress_bar((svo_position + 1) / nb_frames * 100, 30)

                # Check if we have reached the end of the video
                if svo_position >= (nb_frames - 1):  # End of SVO
                    print("\nSVO end has been reached. Exiting now.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if video_writer is not None:
            video_writer.release()
        zed.close()

    print("Output written to: {0}".format(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
