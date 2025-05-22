import csv
import cv2
from datetime import datetime
import time
import os
import glob
import pandas as pd

chunkList = []

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
    return

def OBSCreate(path):
    # Load the CSV file
    filePath = f"{path}/OBS*.csv"
    matchingPath = glob.glob(path)
    createOBSStamps(matchingPath[0])
    stampsPath = glob.glob(f"{matchingPath[0]}/OBStimestamp*")[0]
    print(stampsPath)
    # Read just the first row
    with open(stampsPath, 'r') as f:
        first_line = f.readline().strip()

    # Try to convert first line to an integer — if it fails, it's probably a header
    try:
        int(first_line)
        has_header = False
    except ValueError:
        has_header = True

    if not has_header:
        # Read whole file with no header, add column name, and save
        df = pd.read_csv(stampsPath, header=None, names=["OBS Stamps"])
        df.to_csv(stampsPath, index=False)
        print("Header added.")
    else:
        df = pd.read_csv(stampsPath)
        print("File already has a header.")

    # Save the modified DataFrame back to a CSV file
    df.to_csv(stampsPath, index=False)
    dataLists = ["final_sync", "PDS_sync", "trakstar_sync", "sw_L_acc", "sw_L_gyro", "sw_R_acc", "sw_R_gyro", "zed_sync"]
    for name in dataLists:
        try:
            df.to_csv(f"{path}/Synched Data/{name}.csv", index=False)
        except:
            continue
    print(f"Timestamps have been converted to nano and saved to {filePath}")

if __name__ == "__main__":

    #PUT PATH TO BOOTCAMP DATA HERE
    path = "./data"
    trialFolders = glob.glob(f"{path}/*")
    for trial in trialFolders:
        try:
            print(trial)
            OBSCreate(trial)
        except Exception as e:
            with open("log.txt", 'a', newline='') as logfile:
                logfile.write(f"Exception at {trial}: {e}\n")
                logfile.close()
