import csv
import os

import pyzed.sl as sl
import numpy as np
import cv2

def main():
    trial_path = ".\InguinalH_S112_T1_2024-07-18" # "TRIAL_FOLDER_PATH_HERE"
    zed_frames_path = os.path.join(trial_path, "camera", "Zed", "frames")
    zed_files = [os.path.join(zed_frames_path, x) for x in sorted(os.listdir(zed_frames_path))]

    for i, zfile in enumerate(zed_files):
        print("Processing ", zfile)
        z_csv_path = os.path.join(os.path.dirname(zed_frames_path), os.path.basename(zfile).split('.')[0]+".csv")
        frame_id = 0
        zed = sl.Camera()

        # Set SVO path for playback
        input_path = zfile
        init_parameters = sl.InitParameters()
        init_parameters.set_from_svo_file(input_path)

        # Open the ZED
        zed = sl.Camera()
        err = zed.open(init_parameters)

        svo_image = sl.Mat()

        with open(z_csv_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['ID','TIME'])
            while True:
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    frame_id = zed.get_svo_position()
                    timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                    csv_writer.writerow([frame_id, timestamp])

                elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break


main()
