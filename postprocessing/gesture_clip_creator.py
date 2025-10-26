import os
import cv2
import pandas as pd
from datetime import timedelta
from typing import Optional, Sequence
import numpy as np

def _sec_to_hms(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))

def create_gesture_clips(
    annotation_csv,
    video_path,
    output_dir,
    surgeon_id,
    dry_run=False,
    return_segments=False,
    skip_existing=True,
    write_stats=True,
    stats_csv_name="gesture_clip_stats.csv",
    stats_append=False,          # NEW: append to stats CSV instead of overwriting
    neg1_exclude_cols: Optional[Sequence[str]] = ("obs_frame_idx","seq","server_time", 'frame_number', 'timestamp', 'trakstar_time_ns','trakstar_delta_ns','sw_left_time_ns', 'sw_left_delta_ns', 'sw_right_time_ns', 'sw_right_delta_ns', 'raven_console_time_ns', 'raven_console_delta_ns',
    "Pedal 1 Pressure","Pedal 2 Pressure","Pedal 3 Pressure","Pedal 4 Pressure","Pedal 6 Pressure","Pedal 7 Pressure"),
    drop_neg1: bool = True,
    ignore_classes: Optional[Sequence[int]] = ["Idle",0]
):
    """
    Returns:
      - segments DF if return_segments=True (else None)
      - ALWAYS returns a per-surgeon stats DF (even if write_stats=False) so you can aggregate later.
    """
    import pandas as pd

    # ---------- Read & validate annotations ----------
    ann = pd.read_csv(annotation_csv)
    if 'gesture_code' not in ann.columns:
        raise ValueError("CSV must contain a 'gesture_code' column.")

    candidate_cols = ['obs_frame_idx']
    frame_col = next((c for c in candidate_cols if c in ann.columns), None)
    if frame_col is None:
        raise ValueError(f"CSV must contain one of the frame index columns: {candidate_cols}")
    
        
    if drop_neg1:
        # print(f"Dropping -1 rows for file: {annotation_csv}")
        # ---- Drop rows with sentinel (-1) ----
        # Config:
        drop_mode = "leading"  # "leading" -> drop only initial bad run; "all" -> drop every bad row anywhere
        treat_nan_as_missing = False
        exclude_cols = set(neg1_exclude_cols or ())

        # Pick columns to check (include numeric & object; exclude IDs/timestamps etc.)
        check_cols = [c for c in ann.columns if c not in exclude_cols]
        if check_cols:
            # Coerce to numeric so strings like "-1" are caught; (no ann mutation)
            vals = ann[check_cols].apply(pd.to_numeric, errors="coerce")

            bad_cell = vals.eq(-1)
            if treat_nan_as_missing:
                bad_cell = bad_cell | vals.isna()

            row_is_bad = bad_cell.any(axis=1)  # “bad” row if ANY checked column is -1 (or NaN if enabled)

            total_bad = int(row_is_bad.sum())
            # print(f"Total rows with -1 in checked columns: {total_bad} / {len(ann)}")

            if drop_mode == "leading":
                # count how many consecutive bad rows from the very start
                arr = row_is_bad.to_numpy()
                if arr.size == 0:
                    leading_bad = 0
                elif not arr[0]:
                    leading_bad = 0
                else:
                    # first clean index (first False). If none, drop all.
                    idx = np.nonzero(~arr)[0]
                    leading_bad = int(idx[0]) if idx.size > 0 else len(arr)

                # print(f"Leading bad rows at file start: {leading_bad}")
                if leading_bad > 0:
                    ann = ann.iloc[leading_bad:].reset_index(drop=True)

            elif drop_mode == "all":
                before = len(ann)
                ann = ann.loc[~row_is_bad].reset_index(drop=True)
                print(f"Dropped {before - len(ann)} rows containing -1 anywhere.")

            else:
                raise ValueError("drop_mode must be 'leading' or 'all'")
            # print(f"After dropping rows: {len(ann)}")
    else:
        print(f"Not dropping -1 rows for file: {path}")

    # ---- Exclude ignored gesture classes ----
    if ignore_classes is not None:
        ann = ann[~ann['gesture_code'].isin(ignore_classes)].copy()

    # ann = ann[ann['gesture_code'] != 0].copy()
    if ann.empty:
        print(f"[info] {surgeon_id}: No non-zero gestures found.")
        # write empty stats file if desired
        per_surgeon_stats = pd.DataFrame(columns=[
            "surgeon_id","gesture_code","num_clips","total_frames","total_duration_sec","total_duration_hms"
        ])
        if write_stats:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, stats_csv_name)
            header_needed = not (stats_append and os.path.exists(path))
            per_surgeon_stats.to_csv(path, index=False, mode=('a' if stats_append else 'w'), header=header_needed)
            print(f"[stats] wrote (empty) per-gesture stats to: {path}")
        return (per_surgeon_stats if return_segments else None), per_surgeon_stats

    ann = ann.sort_values(by=frame_col).reset_index(drop=True)
    ann[frame_col] = ann[frame_col].astype(int)

    # ---------- Open video ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------- Compute contiguous gesture segments ----------
    g = ann['gesture_code']
    f = ann[frame_col]
    run_id = ((g != g.shift(1)) | (f != (f.shift(1) + 1))).cumsum()

    segments = (
        ann.assign(run_id=run_id)
           .groupby('run_id')
           .agg(gesture_code=('gesture_code','first'),
                start_frame=(frame_col,'min'),
                end_frame=(frame_col,'max'))
           .reset_index(drop=True)
    )

    segments['start_frame'] = segments['start_frame'].clip(lower=0, upper=max(0, total_frames - 1)).astype(int)
    segments['end_frame']   = segments['end_frame'].clip(lower=0, upper=max(0, total_frames - 1)).astype(int)
    segments = segments[segments['end_frame'] >= segments['start_frame']].copy()
    segments['num_frames']  = segments['end_frame'] - segments['start_frame'] + 1
    segments['duration_sec'] = segments['num_frames'] / float(fps)

    # ---------- Build per-surgeon stats DF ----------
    per_surgeon_stats = (
        segments.groupby('gesture_code')
                .agg(num_clips=('gesture_code','count'),
                     total_frames=('num_frames','sum'),
                     total_duration_sec=('duration_sec','sum'))
                .reset_index()
    )
    per_surgeon_stats.insert(0, 'surgeon_id', surgeon_id)
    per_surgeon_stats['total_duration_hms'] = per_surgeon_stats['total_duration_sec'].apply(_sec_to_hms)

    # ---------- Write stats CSV (optional) ----------
    if write_stats:
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, stats_csv_name)
        header_needed = not (stats_append and os.path.exists(stats_path))
        per_surgeon_stats.to_csv(
            stats_path, index=False,
            mode=('a' if stats_append else 'w'),
            header=header_needed
        )
        print(f"[stats] wrote per-gesture stats to: {stats_path} "
              f"({'append' if stats_append else 'overwrite'})")

    # ---------- Dry run? ----------
    if dry_run:
        print(f"\n[DRY RUN] {surgeon_id} | {os.path.basename(video_path)} "
              f"| FPS={fps:.3f}, Size=({width},{height}), TotalFrames={total_frames}")
        if segments.empty:
            print("(no valid segments after bounds checking)")
        else:
            print(f"{'idx':>4} | {'gesture':>7} | {'start':>8} | {'end':>8} | {'num_frames':>10} | {'dur(s)':>8}")
            print("-" * 70)
            for idx, r in segments.iterrows():
                if surgeon_id.startswith('S'):
                    print(f"{idx:>4} | {int(r['gesture_code']):>7} | {int(r['start_frame']):>8} | "
                        f"{int(r['end_frame']):>8} | {int(r['num_frames']):>10} | {r['duration_sec']:>8.3f}")
                else:
                    print(f"{idx:>4} | {r['gesture_code']:>7} | {int(r['start_frame']):>8} | "
                        f"{int(r['end_frame']):>8} | {int(r['num_frames']):>10} | {r['duration_sec']:>8.3f}")

    # ---------- Real clip writing ----------
    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for _, row in segments.iterrows():
        if surgeon_id.startswith('S'):
            code, s, e = int(row['gesture_code']), int(row['start_frame']), int(row['end_frame'])
        else:
            code, s, e = row['gesture_code'], int(row['start_frame']), int(row['end_frame'])
        expected = int(row['num_frames'])
        gesture_dir = os.path.join(output_dir, str(code))
        os.makedirs(gesture_dir, exist_ok=True)

        outfile = os.path.join(gesture_dir, f"{surgeon_id}_{code}_{s}_{e}.mp4")

        if skip_existing and os.path.exists(outfile):
            print(f"[skip] exists: {outfile}")
            continue

        if dry_run:
            print(f"[dry ] would write: {outfile} ({expected} frames)")
            cap.release()
            break

        writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[warn] could not open writer for {outfile}, skipping.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        written, current = 0, s
        while current <= e:
            ok, frame = cap.read()
            if not ok:
                print(f"[warn] early EOF at frame {current} while writing {outfile}")
                break
            writer.write(frame)
            written += 1
            current += 1
        writer.release()

        if written != expected:
            print(f"[warn] {outfile}: wrote {written}, expected {expected}.")
        else:
            print(f"[ok]  {outfile} ({written} frames)")


    cap.release()
    return (segments if return_segments else None), per_surgeon_stats



if __name__ == "__main__":
    dataset_root_dir = '/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/Processed/'
    output_dir = '/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/Processed/Gesture_Clips/'

    # dataset_root_dir = '/standard/UVA-DSA/MIDAS/Organized/final_data/'
    # output_dir = '/standard/UVA-DSA/MIDAS/Organized/final_data/Gesture_Clips/'

    dry_run = False  # set to True to skip actual writing; just print what would be done

    all_stats = []  # collect per-surgeon stats frames here

    for surgeon in os.listdir(dataset_root_dir):
        surgeon_dir = os.path.join(dataset_root_dir, surgeon)
        if not os.path.isdir(surgeon_dir) or not surgeon.startswith('S'):
        # if not os.path.isdir(surgeon_dir) or not surgeon.startswith('t'):
            continue

        # process only t1 for testing
        # if surgeon != 't1':
        #     continue

        # if surgeon != 'S105_T1':
        #     continue

        print(f"[info] processing {surgeon}")

        annotation_csv = os.path.join(surgeon_dir, "synched_data", f'final_annotation_{surgeon}.csv')
        video_path     = os.path.join(surgeon_dir, "synched_data", f'{surgeon}.mp4')

        print("---------------START------------------")
        print(f"Processing surgeon: {surgeon}\n  annotations: {annotation_csv}\n  video: {video_path}")

        # dry run is fine; we still get per-surgeon stats
        _, per_surgeon_stats = create_gesture_clips(
            annotation_csv=annotation_csv,
            video_path=video_path,
            output_dir=output_dir,
            surgeon_id=surgeon,
            dry_run=dry_run,
            return_segments=False,
            skip_existing=True,
            write_stats=False,                 # we’ll write combined files ourselves
            stats_csv_name="bootcamp_gesture_clip_stats_by_surgeon.csv",
            stats_append=True,                # ignored since write_stats=False
        )
        all_stats.append(per_surgeon_stats)

        print("-------------DONE-----------------")

    # ----- write combined stats -----
    if len(all_stats) and dry_run:
        by_surgeon = pd.concat(all_stats, ignore_index=True)

        overall = (by_surgeon
                   .groupby('gesture_code', as_index=False)
                   .agg(num_clips=('num_clips','sum'),
                        total_frames=('total_frames','sum'),
                        total_duration_sec=('total_duration_sec','sum')))
        from datetime import timedelta
        overall['total_duration_hms'] = overall['total_duration_sec'].apply(lambda s: str(timedelta(seconds=round(s))))
        overall_path = "bootcamp_gesture_clip_stats_overall.csv"
        overall.to_csv(overall_path, index=False)

        print(f"[stats] wrote: {overall_path}")
    else:
        print("[stats] no surgeons produced stats.")
