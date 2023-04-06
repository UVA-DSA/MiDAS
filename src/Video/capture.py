import cv2  
from queue import Queue
from time import time_ns
import csv

def send_vid_data(q, path):

    # Create a CSV file and write the header row
    csv_file = open(path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame #','Computer Time'])

    cap = cv2.VideoCapture(1) #1 for connection to the video capture card
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True: #making sure capture was succesful
            frame_num += 1
            time = time_ns()
            data = [frame_num, time, frame]
            
            csv_writer.writerow(data)
            q.put(data)
            if q.full():
                _ = q.get() #removes last object from q to keep only a certain amount

    