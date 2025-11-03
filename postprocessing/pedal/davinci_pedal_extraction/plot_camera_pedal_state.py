import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# --------------------- USER CONFIG ---------------------------
# ============================================================
CSV_PATH   = "./out_color_changes/frame_metrics.csv"   # per-frame CSV from the processor
OUT_DIR    = "./out_color_changes/plots"                         # where to save plots

# X-axis: "time" uses 'time_s' (if available), "frame" uses 'frame_idx'
X_AXIS     = "time"   # "time" or "frame"

# Which ROIs to plot; None or [] = plot all available
ROIS       = []       # e.g., ["Left_LED", "Right_LED"] or []

# Which columns to plot as y-series.
# You can mix raw metrics and derived columns:
#   - "mean_R", "mean_G", "mean_B", "mean_V"
#   - "mean_V_baseline", "mean_V_delta"
#   - "changed" (binary)
VARS       = [ "changed"]

# Optional: apply simple exponential smoothing to numeric series
USE_EMA    = False
EMA_ALPHA  = 0.15     # 0<alpha<=1 (higher = more smoothing)
# ============================================================


def ensure_out_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ema(series: pd.Series, alpha: float) -> pd.Series:
    """Simple EMA for numeric series; preserves NaNs."""
    s = series.copy()
    mask = ~s.isna()
    if not mask.any():
        return s
    vals = s[mask].values
    out = np.empty_like(vals, dtype=float)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    s.loc[mask] = out
    return s


def pick_x_axis(df: pd.DataFrame, x_axis: str) -> tuple[pd.Series, str]:
    if x_axis.lower() == "time" and "time_s" in df.columns and not df["time_s"].isna().all():
        return df["time_s"], "Time (s)"
    else:
        return df["frame_idx"], "Frame"


def plot_roi_group(df_roi: pd.DataFrame, roi_name: str, vars_to_plot: list[str], out_dir: str, x_axis: str):
    x, x_label = pick_x_axis(df_roi, x_axis)

    plt.figure(figsize=(10, 4.8))

    # Separate numeric vs binary "changed" for nicer styling
    numeric_vars = []
    binary_vars = []
    for v in vars_to_plot:
        if v not in df_roi.columns:
            print(f"[{roi_name}] Skipping missing column: {v}")
            continue
        if v == "changed":
            binary_vars.append(v)
        else:
            # try to treat as numeric
            if pd.api.types.is_numeric_dtype(df_roi[v]):
                numeric_vars.append(v)
            else:
                print(f"[{roi_name}] Skipping non-numeric column: {v}")

    # Plot numeric lines
    for v in numeric_vars:
        y = df_roi[v]
        if USE_EMA:
            y = ema(y, EMA_ALPHA)
        plt.plot(x, y, label=v, linewidth=1.6)

    # Overlay binary "changed" as a stepped line on a secondary y-axis (0/1)
    ax = plt.gca()
    ax2 = None
    if binary_vars:
        ax2 = ax.twinx()
        for v in binary_vars:
            yb = df_roi[v].astype(float)
            ax2.step(x, yb, where="post", linewidth=1.2, alpha=0.6, label=v)
        ax2.set_ylabel("changed (0/1)")
        ax2.set_ylim(-0.1, 1.1)

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax.legend(loc="best")

    plt.title(f"ROI: {roi_name}")
    plt.xlabel(x_label)
    if numeric_vars:
        plt.ylabel("value")
    plt.tight_layout()

    out_path = Path(out_dir, f"{roi_name}_timeseries.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ensure_out_dir(OUT_DIR)

    df = pd.read_csv(CSV_PATH)
    required = {"roi_name", "frame_idx"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")

    # Determine which ROIs to plot
    all_rois = list(df["roi_name"].dropna().unique())
    rois = all_rois if not ROIS else [r for r in ROIS if r in all_rois]
    missing = [r for r in ROIS if r not in all_rois]
    if missing:
        print(f"Warning: requested ROIs not found in CSV: {missing}")
    if not rois:
        print("No matching ROIs found. Nothing to plot.")
        return

    # Sort by x-axis for consistent plotting
    if X_AXIS.lower() == "time" and "time_s" in df.columns:
        df = df.sort_values(["roi_name", "time_s", "frame_idx"])
    else:
        df = df.sort_values(["roi_name", "frame_idx"])

    # Plot each ROI separately
    for r in rois:
        g = df[df["roi_name"] == r].copy()
        plot_roi_group(g, r, VARS, OUT_DIR, X_AXIS)

    print("Done.")


if __name__ == "__main__":
    main()
