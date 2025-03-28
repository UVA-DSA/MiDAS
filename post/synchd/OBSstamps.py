import csv
import cv2
from datetime import datetime
import time

date_str = '2024-07-16 14:39:37'  # 'YYYY-MM-DD HH:MM:SS'

# Create a datetime object
dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

epoch_time = int(time.mktime(dt.timetuple()))*1000 +  500
epoch_time = 1721334965000000000
file_name = "OBStimestamp_Bowel_S207_T2_2024-07-18"

video_path = "C:/Users/Zachary/Documents/Git Repos/DataCollectionSystem/data/Bowel_S207_T2_2024-07-18/obs/2024-07-18 16-36-07.mkv"
cap = cv2.VideoCapture(video_path)

# Get total frame count
expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()



with open(file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile) 
    j = 0
    for i in range (1, expected_frames + 1):
        csvwriter.writerow([epoch_time + j])
        epoch_time += j
        j = 33333333
        if (i % 3 == 0):
            j = 33333334

csvfile.close()

 








