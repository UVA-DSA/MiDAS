


ROOT_PATH = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/Processed/"
TRIALS = [    
    "S105_T1","S106_T1","S106_T2","S112_T1","S112_T2","S116_T1","S116_T2","S116_T4","S116_T5",
    "S118_T1","S200_T1","S201_T1","S201_T2","S202_T1","S203_T1","S204_T1","S209_T2","S210_T1",
    "S214_T1","S214_T4","S214_T6","S215_T3","S215_T4","S217_T2","S217_T3","S217_T4","S218_T1","S219_T1",
]

def visualize_pedals_on_video(video_path: str,
                              gt_camera_pedal_csv: str,
                              gt_armswap_pedal_csv: str,
                              gt_energy_pedals_csv: str,
                              output_path: str):
    # Placeholder for the actual implementation of visualization.
    # This function would read the video and the CSV files,
    # overlay the pedal states on the video frames, and save the output video.
    pass



for trial in TRIALS:
    video_path = f"{ROOT_PATH}/{trial}/synched_data/{trial}.mp4"
    out_dir = f"{ROOT_PATH}/{trial}/synched_data/{trial}_pedal_visualized.mp4"

    gt_camera_pedal_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_camera_pedal_gt_hamid.csv"
    gt_armswap_pedal_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_arm_swap_pedal_gt.csv"
    gt_energy_pedals_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_energy_pedal_gt.csv"

    print("*-*-"*20)
    print(f"Visualizing pedals for trial {trial}...")

    visualize_pedals_on_video(
        video_path=video_path,
        gt_camera_pedal_csv=gt_camera_pedal_csv,
        gt_armswap_pedal_csv=gt_armswap_pedal_csv,
        gt_energy_pedals_csv=gt_energy_pedals_csv,
        output_path=out_dir
    )


