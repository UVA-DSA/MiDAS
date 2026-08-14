"""Run the 6D hand pose extraction over many SVO recordings in one go.

Accepts any mix of files, directories and glob patterns, so it does not assume
any particular dataset layout. Each recording is processed in its own
subprocess: a failure on one file (a corrupt recording, a ZED SDK crash) does
not abort the rest of the batch.

Options this script does not recognise are forwarded verbatim to
``save_6d_palm.py``, so every tracking and camera switch is available here too::

    python batch_process_svo.py /data --frame-skip 2 --depth-mode NEURAL

Run ``python batch_process_svo.py --help`` for the full list, and
``python zed/save_6d_palm.py --help`` for the options that can be forwarded.
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKER = PACKAGE_DIR / "zed" / "save_6d_palm.py"

#: Recording extensions searched for inside directories.
DEFAULT_EXTENSIONS = (".svo2", ".svo")


def find_recordings(inputs, extensions, pattern=None, recursive=True):
    """Expand `inputs` into a list of ``(recording, search_root)`` pairs.

    Each input may be a single file, a directory to search, or a glob pattern.
    `search_root` is the directory the recording was discovered under and is
    used to mirror the input tree into the output directory; it is ``None``
    for recordings named individually.
    """
    found = []
    seen = set()

    def add(path, root):
        resolved = Path(path).resolve()
        if resolved in seen or not resolved.is_file():
            return
        seen.add(resolved)
        found.append((resolved, root))

    for item in inputs:
        path = Path(item)

        if path.is_file():
            add(path, None)
            continue

        if path.is_dir():
            root = path.resolve()
            matches = []
            if pattern:
                matches = sorted(root.glob(pattern))
            else:
                prefix = "**/*" if recursive else "*"
                for extension in extensions:
                    matches.extend(sorted(root.glob(prefix + extension)))
            if not matches:
                print("No recordings found in: {0}".format(path))
            for match in sorted(matches):
                add(match, root)
            continue

        # Not an existing path: treat it as a glob pattern.
        matches = sorted(glob.glob(str(item), recursive=True))
        if not matches:
            print("Warning: no file matches '{0}'".format(item))
        for match in matches:
            add(match, None)

    return found


def output_path_for(recording, root, output_dir, flat):
    """Where the CSV for `recording` should be written, or ``None`` for the default."""
    if output_dir is None:
        return None

    output_dir = Path(output_dir)
    name = recording.stem + ".csv"

    if flat or root is None:
        return output_dir / name

    try:
        relative = recording.relative_to(root).parent
    except ValueError:
        relative = Path()
    return output_dir / relative / name


def build_parser():
    parser = argparse.ArgumentParser(
        prog="batch_process_svo.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # every recording under a directory tree, CSVs written next to each file
  python batch_process_svo.py /data/session01

  # collect the CSVs elsewhere, mirroring the input folder structure
  python batch_process_svo.py /data -o /results

  # only the recordings matching a layout of your own
  python batch_process_svo.py /data --pattern "*/session*/frames/*.svo2"

  # explicit files, plus options forwarded to save_6d_palm.py
  python batch_process_svo.py a.svo2 b.svo2 --frame-skip 2 --max-hands 1

  # see what would run, without running it
  python batch_process_svo.py /data --dry-run
""",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="SVO files, directories to search, or glob patterns.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory for the CSV files. Default: each CSV is written next "
             "to its recording.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write every CSV directly into --output-dir instead of mirroring "
             "the input folder structure.",
    )
    parser.add_argument(
        "--pattern",
        help="Glob pattern applied inside each input directory, e.g. "
             "\"*/session*/*.svo2\". Overrides --extensions and --no-recursive.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma separated recording extensions to search for "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only look in the top level of each input directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip recordings whose output CSV already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch as soon as one recording fails.",
    )
    parser.add_argument(
        "--capture-output",
        action="store_true",
        help="Hide the per-file output and only show it when a file fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the commands that would run, then exit.",
    )
    parser.add_argument(
        "--worker",
        default=str(DEFAULT_WORKER),
        help="Path to the processing script (default: the bundled save_6d_palm.py).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run the worker (default: the current one).",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)

    worker = Path(args.worker)
    if not worker.is_file():
        print("Error: worker script not found: {0}".format(worker), file=sys.stderr)
        return 2

    extensions = tuple(
        ext if ext.startswith(".") else "." + ext
        for ext in (e.strip() for e in args.extensions.split(","))
        if ext
    )

    recordings = find_recordings(
        args.inputs,
        extensions=extensions,
        pattern=args.pattern,
        recursive=not args.no_recursive,
    )

    if not recordings:
        print("No recordings to process.")
        return 1

    print("Found {0} recording(s) to process.".format(len(recordings)))
    if forwarded:
        print("Forwarding to {0}: {1}".format(worker.name, " ".join(forwarded)))
    print("=" * 60)

    succeeded, failed, skipped = [], [], []

    for index, (recording, root) in enumerate(recordings, 1):
        output = output_path_for(recording, root, args.output_dir, args.flat)

        if args.skip_existing:
            existing = output or recording.with_suffix(".csv")
            if Path(existing).is_file():
                print("[{0}/{1}] Skipping (output exists): {2}".format(
                    index, len(recordings), recording))
                skipped.append(recording)
                continue

        command = [args.python, str(worker), str(recording)]
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            command += ["-o", str(output)]
        command += forwarded

        print("\n[{0}/{1}] {2}".format(index, len(recordings), recording))
        if args.dry_run:
            print("  would run: {0}".format(" ".join(command)))
            continue

        try:
            result = subprocess.run(command, capture_output=args.capture_output, text=True)
        except OSError as exc:
            print("  [FAIL] could not start the worker: {0}".format(exc))
            failed.append((recording, str(exc)))
            if args.stop_on_error:
                break
            continue

        if result.returncode == 0:
            print("  [OK]")
            succeeded.append(recording)
        else:
            print("  [FAIL] exit code {0}".format(result.returncode))
            if args.capture_output:
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
            failed.append((recording, "exit code {0}".format(result.returncode)))
            if args.stop_on_error:
                print("Stopping because --stop-on-error was given.")
                break

    if args.dry_run:
        print("\nDry run: nothing was processed.")
        return 0

    print("\n" + "=" * 60)
    print("Summary")
    print("  Processed:  {0}".format(len(succeeded)))
    print("  Skipped:    {0}".format(len(skipped)))
    print("  Failed:     {0}".format(len(failed)))
    for recording, reason in failed:
        print("    - {0} ({1})".format(recording, reason))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
