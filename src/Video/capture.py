import cv2  
from queue import Queue
from time import time_ns
import csv
import os

def send_vid_data(q, path):

    # Create a CSV file and write the header row
    csv_path = path + '/video.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame #','Computer Time'])

    img_dir = path + '/imgs'
        
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    
    cap = cv2.VideoCapture(0) #1 for connection to the video capture card
    frame_num = 0
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret == True: #making sure capture was succesful
                frame_num += 1
                time = time_ns()
                data = [frame_num, time, frame]
                
                filename = f"/{frame_num}_{time}.jpeg"
                img_path = img_dir + filename
                cv2.imwrite(img_path, frame)

                csv_writer.writerow(data)
                q.put(data)
                if q.full():
                    _ = q.get() #removes last object from q to keep only a certain amount

        except KeyboardInterrupt:
            exit(0)