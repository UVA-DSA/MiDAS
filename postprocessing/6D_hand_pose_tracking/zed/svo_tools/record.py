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

"""Record an SVO file from a connected ZED camera.

Adapted from the Stereolabs SVO recording sample: every capture setting is a
command line option and the recording stops cleanly on Ctrl+C.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pyzed.sl as sl

from HandTrackingModule.Zed import (
    DEPTH_MODE_CHOICES,
    RESOLUTION_CHOICES,
    resolve_depth_mode,
    resolve_enum,
    resolve_resolution,
)

COMPRESSION_CHOICES = ("H264", "H265", "LOSSLESS", "LOSSY")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="record.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python record.py /data/session01/take1.svo2
  python record.py take1.svo2 --resolution HD1080 --fps 30 --compression H265
  python record.py take1.svo2 --max-frames 900        # stop after 900 frames
""",
    )
    parser.add_argument("output", help="Path of the SVO file to create.")
    parser.add_argument(
        "--resolution",
        default="HD720",
        choices=RESOLUTION_CHOICES,
        help="Recording resolution (default: %(default)s).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frame rate (default: %(default)s).",
    )
    parser.add_argument(
        "--depth-mode",
        default="QUALITY",
        choices=DEPTH_MODE_CHOICES,
        help="Depth mode used while recording (default: %(default)s).",
    )
    parser.add_argument(
        "--compression",
        default="H264",
        choices=COMPRESSION_CHOICES,
        help="SVO compression mode (default: %(default)s).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop automatically after this many frames. Default: record until Ctrl+C.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    directory = os.path.dirname(os.path.abspath(args.output))
    if directory:
        os.makedirs(directory, exist_ok=True)

    init = sl.InitParameters()
    try:
        init.camera_resolution, _ = resolve_resolution(args.resolution)
        init.depth_mode, _ = resolve_depth_mode(args.depth_mode)
        compression, _ = resolve_enum(
            sl.SVO_COMPRESSION_MODE, (args.compression,), "SVO compression mode"
        )
    except ValueError as exc:
        print("Error: {0}".format(exc), file=sys.stderr)
        return 2
    init.camera_fps = args.fps

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Unable to open the ZED camera: {0}".format(repr(status)), file=sys.stderr)
        return 1

    recording_param = sl.RecordingParameters(args.output, compression)
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Unable to start recording: {0}".format(repr(err)), file=sys.stderr)
        cam.close()
        return 1

    runtime = sl.RuntimeParameters()
    print("Recording to {0}. Press Ctrl+C to stop.".format(args.output))
    frames_recorded = 0
    try:
        while True:
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                frames_recorded += 1
                print("Frame count: {0}".format(frames_recorded), end="\r")
                if args.max_frames is not None and frames_recorded >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.disable_recording()
        cam.close()

    print("\nRecorded {0} frames to {1}".format(frames_recorded, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
