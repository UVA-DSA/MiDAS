import csv
import cv2
from datetime import datetime
import time
import os
import glob

def createOBSStamps(mainPath):
    filePath = f"{mainPath}/obs/*.mkv"
    video_path = glob.glob(filePath)[0]
    # Extract filename
    filename = os.path.basename(video_path)

    # Remove extension
    timestamp_str = os.path.splitext(filename)[0]  # "2024-07-18 15-48-29"

    # Convert to datetime object
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H-%M-%S")

    epoch_time = int(time.mktime(dt.timetuple()))*1000000000
    file_name = f"{mainPath}/OBStimestamp_{os.path.basename(mainPath)}.csv"

    cap = cv2.VideoCapture(video_path)
    # Get total frame count
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frame_duration_ns = int((1 / frame_rate) * 1_000_000_000)

    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        current_time = epoch_time
        for i in range(expected_frames):
            csvwriter.writerow([current_time])
            current_time += frame_duration_ns

    csvfile.close()

    








