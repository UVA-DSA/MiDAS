import os
from typing import Optional, Tuple
import pandas as pd



ROOT_PATH = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/Processed/"
TRIALS = [    
    "S105_T1","S106_T1","S106_T2","S112_T1","S112_T2","S116_T1","S116_T2","S116_T4","S116_T5",
    "S118_T1","S200_T1","S201_T1","S201_T2","S202_T1","S203_T1","S204_T1","S209_T2","S210_T1",
    "S214_T1","S214_T4","S214_T6","S215_T3","S215_T4","S217_T2","S217_T3","S217_T4","S218_T1","S219_T1",
]

def merge_pedals(
    gt_camera_pedal_csv: str,
    gt_armswap_pedal_csv: str,
    gt_energy_pedals_csv: str,
    output_csv: str
):
  
    # check paths exist
    if not os.path.isfile(gt_camera_pedal_csv):
        print(f"Error: {gt_camera_pedal_csv} does not exist.")
        return
    if not os.path.isfile(gt_armswap_pedal_csv):
        print(f"Error: {gt_armswap_pedal_csv} does not exist.")
        return
    if not os.path.isfile(gt_energy_pedals_csv):
        print(f"Error: {gt_energy_pedals_csv} does not exist.")
        return
    
    gt_camera_df = pd.read_csv(gt_camera_pedal_csv)
    gt_armswap_df = pd.read_csv(gt_armswap_pedal_csv)
    gt_energy_df = pd.read_csv(gt_energy_pedals_csv)


    # first check if there is a `Frame Num` column in all dataframes and if so rename it to 'obs_frame_idx'
    if 'Frame Num' in gt_camera_df.columns:
        gt_camera_df = gt_camera_df.rename(columns={'Frame Num': 'obs_frame_idx'})
    if 'Frame Num' in gt_armswap_df.columns:
        gt_armswap_df = gt_armswap_df.rename(columns={'Frame Num': 'obs_frame_idx'})
    if 'Frame Num' in gt_energy_df.columns:
        gt_energy_df = gt_energy_df.rename(columns={'Frame Num': 'obs_frame_idx'})

    print(gt_camera_df.head(1))
    print(gt_armswap_df.head(1))
    print(gt_energy_df.head(1))


    pass



for trial in TRIALS:

    gt_camera_pedal_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_camera_pedal_gt_hamid.csv"
    gt_armswap_pedal_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_arm_swap_pedal_gt.csv"
    gt_energy_pedals_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_primary_secondary_pedals_gt.csv"

    out_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_pds_gt.csv"

    print("*-*-"*20)
    print(f"Merging pedals for trial {trial}...")

    merge_pedals(
        gt_camera_pedal_csv=gt_camera_pedal_csv,
        gt_armswap_pedal_csv=gt_armswap_pedal_csv,
        gt_energy_pedals_csv=gt_energy_pedals_csv,
        output_csv=out_csv
    )