import csv
import os
import glob

import pyzed.sl as sl
import numpy as np
import cv2

def extractZed(path):
    trial_path = path #TRIAL_FOLDER_PATH_HERE
    zed_frames_path = os.path.join(trial_path, "camera", "Zed", "frames")
    zed_files = [os.path.join(zed_frames_path, x) for x in sorted(os.listdir(zed_frames_path))]
    z_csv_path = os.path.join(os.path.dirname(zed_frames_path),"zed_frames.csv")
    csv_writer = csv.writer(open(z_csv_path, 'w', newline=''))
    csv_writer.writerow(['Zed Stamps', 'ID'])
    frame_id = 0

    for i, zfile in enumerate(zed_files):
        print("Processing ", zfile)
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
            while True:
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    frame_id = zed.get_svo_position()
                    timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                    csv_writer.writerow([timestamp,f"{i}_{frame_id}"])


                elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break

if __name__ == "__main__":
    paths = glob.glob("D:/Data MIDAS/*")
    Err = []
    for path in paths:
        try:
            extractZed(path)
        except Exception as e:
            Err.append([path, e])
    print(Err)
