"""Download the ZED SDK AI models (neural depth, skeleton, object detection).

The neural depth modes used by this package need these models to be present in
the SDK resources directory. The SDK normally downloads them on first use; this
script fetches them up front, which is handy on machines without internet
access at run time or when the automatic download is blocked.
"""

import argparse
import os
import sys

import requests

BASE_URL = 'http://download.stereolabs.com/ai/model/'

#: Directory the ZED SDK looks in, per platform. Override with --output-dir or
#: the ZED_RESOURCES_DIR environment variable.
DEFAULT_RESOURCE_DIRS = {
    'posix': '/usr/local/zed/resources/',
    'nt': 'C:/ProgramData/Stereolabs/resources/',
}

MODELS_SDK_4 = [
    'objects_performance_3.2.model',
    'objects_medium_3.2.model',
    'objects_accurate_3.2.model',
    'skeleton_body18_3.2.model',
    'skeleton_body38_3.5.model',
    'person_head_performance_2.4.model',
    'person_head_accurate_2.4.model',
    'person_reid_1.4.model',
    'neural_depth_2.0.model',
    'neural_depth_3.6.model',
]

MODELS_SDK_5 = [
    'objects_performance_3.2.model',
    'objects_medium_3.2.model',
    'objects_accurate_3.2.model',
    'skeleton_body18_3.2.model',
    'skeleton_body38_3.5.model',
    'person_head_performance_2.4.model',
    'person_head_accurate_2.4.model',
    'person_reid_1.4.model',
    'neural_depth_5.2.model',
    'neural_depth_light_5.2.model',
]

MIN_MODEL_SIZE = 1024 * 1024  # a valid model file is always larger than 1 MB


def default_output_directory():
    """Resources directory of the installed SDK for the current platform."""
    env_dir = os.environ.get('ZED_RESOURCES_DIR')
    if env_dir:
        return env_dir
    return DEFAULT_RESOURCE_DIRS.get(os.name, './resources/')


def detect_sdk_version():
    """Major version of the installed ZED SDK, or ``None`` if undetectable."""
    try:
        import pyzed.sl as sl
    except ImportError:
        return None
    try:
        version = str(sl.Camera.get_sdk_version())
        return version.split('.')[0]
    except Exception:  # noqa: BLE001 - detection is best effort
        return None


def show_progress_bar(downloaded, total, bar_length=40):
    percent = float(downloaded) / total if total else 0
    arrow = '=' * int(round(percent * bar_length) - 1) + '>' if percent > 0 else ''
    spaces = ' ' * (bar_length - len(arrow))
    mb_downloaded = downloaded / 1024 / 1024
    mb_total = total / 1024 / 1024 if total else 0
    sys.stdout.write(
        "\rProgress: [{0}{1}] {2:.1f}% ({3:.2f}MB/{4:.2f}MB)".format(
            arrow, spaces, percent * 100, mb_downloaded, mb_total
        )
    )
    sys.stdout.flush()


def download_files(base_url, model_list, output_folder, overwrite=False):
    """Download each model into `output_folder`. Returns the failure count."""
    os.makedirs(output_folder, exist_ok=True)
    failures = 0

    for filename in model_list:
        file_path = os.path.join(output_folder, filename)
        if not overwrite and os.path.isfile(file_path) and os.path.getsize(file_path) >= MIN_MODEL_SIZE:
            print("- Skipping {0} (already present)".format(filename))
            continue

        full_url = base_url.rstrip('/') + '/' + filename.lstrip('/')
        try:
            with requests.get(full_url, stream=True, allow_redirects=True, timeout=30) as response:
                final_response = response.history[-1] if response.history else response
                if final_response.status_code not in (200, 302):
                    print("Failed to download {0} from {1} (status {2})".format(
                        filename, full_url, final_response.status_code))
                    failures += 1
                    continue

                total_length = int(response.headers.get('content-length', 0))
                if total_length < MIN_MODEL_SIZE:
                    print(
                        "Error: {0} is too small ({1:.2f} kB, expected more than 1MB). "
                        "Skipping. Check the model URL or try again later.".format(
                            filename, total_length / 1024)
                    )
                    failures += 1
                    continue

                print("- Downloading {0} from {1} to {2}".format(filename, full_url, file_path))
                downloaded = 0
                with open(file_path, 'wb') as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            handle.write(chunk)
                            downloaded += len(chunk)
                            show_progress_bar(downloaded, total_length)
                sys.stdout.write('\n')
        except requests.exceptions.RequestException as exc:
            print("Error while downloading {0} from {1}: {2}".format(filename, full_url, exc))
            failures += 1
        except OSError as exc:
            print(
                "Error writing {0}: {1}\nThe SDK resources directory usually needs "
                "administrator/root rights; either rerun elevated or pass "
                "--output-dir.".format(file_path, exc)
            )
            failures += 1

    return failures


def print_urls(base_url, model_list):
    for filename in model_list:
        print("{0}: {1}{2}".format(filename, base_url.rstrip('/') + '/', filename.lstrip('/')))


def build_parser():
    parser = argparse.ArgumentParser(
        prog='download_zed_models.py',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python download_zed_models.py                          # detect the SDK version
  python download_zed_models.py --sdk-version 4
  python download_zed_models.py --output-dir ./models/
  python download_zed_models.py --print-url-only

The default output directory is the SDK resources folder for this platform
({0}), which can be overridden with --output-dir or the ZED_RESOURCES_DIR
environment variable. Writing there usually requires administrator rights.
""".format(default_output_directory()),
    )
    parser.add_argument(
        '--sdk-version',
        choices=['4', '5', 'all', 'auto'],
        default='auto',
        help="Which model set to fetch. 'auto' detects the installed SDK and "
             "falls back to 'all' (default: %(default)s).",
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Where to save the models. Default: the SDK resources directory.',
    )
    parser.add_argument(
        '--print-url-only',
        action='store_true',
        help='Print the download URLs instead of downloading the files.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Re-download models that are already present.',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    sdk_version = args.sdk_version
    if sdk_version == 'auto':
        detected = detect_sdk_version()
        if detected in ('4', '5'):
            print("Detected ZED SDK {0}.x".format(detected))
            sdk_version = detected
        else:
            print("Could not detect the ZED SDK version, downloading all models.")
            sdk_version = 'all'

    if sdk_version == '4':
        model_list = MODELS_SDK_4
    elif sdk_version == '5':
        model_list = MODELS_SDK_5
    else:  # all: union, preserving order
        model_list = MODELS_SDK_4 + [m for m in MODELS_SDK_5 if m not in MODELS_SDK_4]

    if args.print_url_only:
        print_urls(BASE_URL, model_list)
        return 0

    output_directory = args.output_dir or default_output_directory()
    print("Saving models to: {0}".format(output_directory))

    try:
        os.makedirs(output_directory, exist_ok=True)
    except OSError as exc:
        print(
            "Error: cannot create {0}: {1}\nUse --output-dir to pick a writable "
            "location, or rerun with administrator/root rights.".format(output_directory, exc),
            file=sys.stderr,
        )
        return 1

    failures = download_files(BASE_URL, model_list, output_directory, overwrite=args.overwrite)
    if failures:
        print("\n{0} model(s) failed to download.".format(failures))
        return 1

    print("\nAll models are in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
