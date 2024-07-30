import csv

from datetime import datetime
import time

date_str = '2024-07-16 14:39:37'  # 'YYYY-MM-DD HH:MM:SS'

# Create a datetime object
dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

epoch_time = int(time.mktime(dt.timetuple()))*1000 +  500
video_length = 325 #seconds
file_name = "OBStimestamp_ESL_S100_T2_2024-07-16"


expected_frames = (video_length + 1) * 30



with open(file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile) 
    j = 0
    for i in range (1, expected_frames + 1):
        csvwriter.writerow([epoch_time + j])
        epoch_time += j
        j = 33
        if (i % 3 == 0):
            j = 34

 








