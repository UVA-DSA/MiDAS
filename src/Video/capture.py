import cv2  
from queue import Queue
from time import time_ns

def send_vid_data(q):
    cap = cv2.VideoCapture(1) #1 for connection to the video capture card
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True: #making sure capture was succesful
            frame_num += 1
            time = time_ns()
            data = [frame, time, frame_num]
            q.put(data)
            if q.full():
                _ = q.get() #removes last object from q to keep only a certain amount

    