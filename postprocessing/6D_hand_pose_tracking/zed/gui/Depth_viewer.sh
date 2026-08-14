#!/usr/bin/env bash
# Launch the ZED_Depth_Viewer tool that ships with the ZED SDK.
#
# The SDK tools directory is auto-detected. Override it if the SDK is
# installed somewhere else:
#     ZED_TOOLS_DIR=/opt/zed/tools ./Depth_viewer.sh
set -euo pipefail

TOOL="ZED_Depth_Viewer"

if [ -n "${ZED_TOOLS_DIR:-}" ]; then
    CANDIDATES=("$ZED_TOOLS_DIR")
else
    CANDIDATES=(
        "/usr/local/zed/tools"
        "/opt/zed/tools"
        "$HOME/zed/tools"
        "/c/Program Files (x86)/ZED SDK/tools"
        "/c/Program Files/ZED SDK/tools"
    )
fi

for dir in "${CANDIDATES[@]}"; do
    for exe in "$dir/$TOOL" "$dir/$TOOL.exe"; do
        if [ -x "$exe" ]; then
            exec "$exe" "$@"
        fi
    done
done

echo "Could not find $TOOL. Set ZED_TOOLS_DIR to your ZED SDK tools directory." >&2
exit 1
