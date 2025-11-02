import os
from typing import Optional
import pandas as pd
from functools import reduce

ROOT_PATH = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/Processed/"
TRIALS = [    
    "S105_T1","S106_T1","S106_T2","S112_T1","S112_T2","S116_T1","S116_T2","S116_T4","S116_T5",
    "S118_T1","S200_T1","S201_T1","S201_T2","S202_T1","S203_T1","S204_T1","S209_T2","S210_T1",
    "S214_T1","S214_T4","S214_T6","S215_T3","S215_T4","S217_T2","S217_T3","S217_T4","S218_T1","S219_T1",
]

import os
from typing import Optional
import pandas as pd
from functools import reduce

import os
from typing import Optional, Tuple
import pandas as pd

def merge_pedals(
    gt_camera_pedal_csv: Optional[str],
    gt_armswap_pedal_csv: Optional[str],
    gt_energy_pedals_csv: Optional[str],
    output_csv: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge any subset of {camera, armswap, energy} pedal CSVs.
    Any path may be None (it will be skipped). At least one source must be provided.
    Returns (final_df_out, stats_df).
    """
    # ---- load what we have ----
    cam = arm = eng = None
    if gt_camera_pedal_csv:
        if not os.path.isfile(gt_camera_pedal_csv):
            raise FileNotFoundError(gt_camera_pedal_csv)
        cam = pd.read_csv(gt_camera_pedal_csv)
    if gt_armswap_pedal_csv:
        if not os.path.isfile(gt_armswap_pedal_csv):
            raise FileNotFoundError(gt_armswap_pedal_csv)
        arm = pd.read_csv(gt_armswap_pedal_csv)
    if gt_energy_pedals_csv:
        if not os.path.isfile(gt_energy_pedals_csv):
            raise FileNotFoundError(gt_energy_pedals_csv)
        eng = pd.read_csv(gt_energy_pedals_csv)

    if cam is None and arm is None and eng is None:
        raise ValueError("No inputs provided: all CSV paths are None.")

    # ---- normalize frame index ----
    def _norm_idx(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return None
        if "Frame Num" in df.columns:
            df = df.rename(columns={"Frame Num": "obs_frame_idx"})
        if "obs_frame_idx" not in df.columns:
            raise ValueError("Missing 'obs_frame_idx' (or 'Frame Num') in one of the CSVs.")
        return df

    cam = _norm_idx(cam)
    arm = _norm_idx(arm)
    eng = _norm_idx(eng)

    # ---- known columns per source ----
    camera_cols = ['Instrument_1_Camera', 'Instrument_2_Camera', 'Instrument_3_Camera', 'Instrument_4_Camera']
    energy_cols = ['Upper Left', 'Upper Right', 'Lower Left', 'Lower Right']
    armswap_cols = ['Arm_Swap'] if (arm is not None and 'Arm_Swap' in arm.columns) else []

    # ---- build the merged DF incrementally ----
    base = None
    def _merge_into(base_df, src_df, keep_cols):
        if src_df is None:
            return base_df
        cols = ['obs_frame_idx'] + [c for c in keep_cols if c in src_df.columns]
        if base_df is None:
            return src_df[cols].copy()
        return pd.merge(base_df, src_df[cols], on='obs_frame_idx', how='outer')

    base = _merge_into(base, cam, camera_cols)
    base = _merge_into(base, arm, armswap_cols if armswap_cols else [])
    base = _merge_into(base, eng, energy_cols)

    if base is None:
        # should not happen because we checked earlier, but guard anyway
        raise ValueError("Nothing to merge after filtering columns.")

    base = base.sort_values('obs_frame_idx').reset_index(drop=True)

    # ---- fill/coerce binary for any present cols ----
    for cols in (camera_cols, energy_cols, armswap_cols):
        for c in cols:
            if c in base.columns:
                base[c] = (pd.to_numeric(base[c], errors='coerce').fillna(0).astype(int) > 0).astype(int)

    # ---- aggregate camera instruments → single Camera_Pedal (OR) ----
    present_cam = [c for c in camera_cols if c in base.columns]
    if present_cam:
        base['Camera_Pedal'] = base[present_cam].max(axis=1).fillna(0).astype(int)
    elif cam is not None:
        # camera CSV present but no instrument cols (unlikely)—fallback to 0
        base['Camera_Pedal'] = 0

    # ---- choose output columns (only those that exist) ----
    keep_cols = ['obs_frame_idx']
    if 'Camera_Pedal' in base.columns:
        keep_cols.append('Camera_Pedal')
    keep_cols += [c for c in energy_cols if c in base.columns]
    keep_cols += [c for c in armswap_cols if c in base.columns]

    final_df_out = base[keep_cols].copy()

    # ---- save merged CSV (optional) ----
    if output_csv:
        # check if csv is already there
        if not os.path.isfile(output_csv):
            final_df_out.to_csv(output_csv, index=False)

    # ---- stats ----
    stats_df = pd.DataFrame(columns=['Pedal_or_Event', 'Event_Count', 'Event_Share_Percent'])

    print("\nA) Per-frame statistics (fraction of frames with each flag = 1):")
    total_frames = len(final_df_out)
    event_cols = [c for c in keep_cols if c != 'obs_frame_idx']
    for c in event_cols:
        pressed = int(pd.to_numeric(final_df_out[c], errors='coerce').fillna(0).astype(int).sum())
        pct = 100.0 * pressed / total_frames if total_frames else 0.0
        print(f"  {c:>20}: {pressed}/{total_frames}  ({pct:6.2f}%)")

    print("\nB) Share of all press/event occurrences (sums to 100%):")
    if not event_cols:
        print("  No pedal or event columns found.")
    else:
        per_col_counts = {c: int(final_df_out[c].sum()) for c in event_cols}
        total_events = sum(per_col_counts.values())
        if total_events == 0:
            print("  No events detected.")
        else:
            rows = []
            for c in event_cols:
                pct = 100.0 * per_col_counts[c] / total_events
                print(f"  {c:>20}: {per_col_counts[c]} events  ({pct:6.2f}%)")
                rows.append((c, per_col_counts[c], round(pct, 2)))
            stats_df = pd.DataFrame(rows, columns=['Pedal_or_Event','Event_Count','Event_Share_Percent'])
            if output_csv:
                stats_csv = output_csv.replace('.csv', '_pedal_event_stats.csv')
                if not os.path.isfile(stats_csv):   
                    stats_df.to_csv(stats_csv, index=False)
                print(f"\n   ✅ Event statistics saved to: {stats_csv}")

    return final_df_out, stats_df


if __name__ == "__main__":
    all_stats = []

    for trial in TRIALS:
        cam_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_camera_pedal_gt_hamid.csv"
        arm_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_arm_swap_pedal_gt.csv"
        eng_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_primary_secondary_pedals_gt.csv"
        out_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_pds_gt.csv"

        print("*-*-"*20)
        print(f"Merging pedals for trial {trial}...")

        merged_df, stats_df = merge_pedals(
            gt_camera_pedal_csv=None,
            gt_armswap_pedal_csv=None,
            gt_energy_pedals_csv=eng_csv,
            output_csv=out_csv
        )

        # tag trial for global aggregation
        if not stats_df.empty:
            stats_df = stats_df.assign(Trial=trial)
        else:
            stats_df = pd.DataFrame(columns=['Pedal_or_Event','Event_Count','Event_Share_Percent','Trial'])
        all_stats.append(stats_df)

    # ---- GLOBAL AGG ----
    if all_stats:
        global_df = pd.concat(all_stats, ignore_index=True)
        

        avg_share = (global_df
                     .groupby('Pedal_or_Event', as_index=False)['Event_Share_Percent']
                     .mean()
                     .rename(columns={'Event_Share_Percent': 'Avg_Event_Share_Percent'}))

        total_counts = (global_df
                        .groupby('Pedal_or_Event', as_index=False)['Event_Count']
                        .sum()
                        .rename(columns={'Event_Count': 'Total_Event_Count'}))

        n_trials = (global_df.drop_duplicates(['Trial','Pedal_or_Event'])
                    .groupby('Pedal_or_Event', as_index=False)['Trial']
                    .count()
                    .rename(columns={'Trial': 'N_Trials_Observed'}))

        global_stats = (avg_share
                        .merge(total_counts, on='Pedal_or_Event', how='outer')
                        .merge(n_trials, on='Pedal_or_Event', how='outer')) \
                        .sort_values('Avg_Event_Share_Percent', ascending=False)

        global_csv = os.path.join(ROOT_PATH, "bootcamp_pedal_event_stats_global.csv")
        global_stats.to_csv(global_csv, index=False)

        print("\n✅ Global pedal/event stats saved to:", global_csv)
        print(global_stats.to_string(index=False))
    else:
        print("No per-trial stats produced; global aggregation skipped.")
