import cv2  
import socket
from time import time_ns
import pickle

def send_vid_data():
    cap = cv2.VideoCapture(1) #1 for connection to the video capture card
    frame_num = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 12345))
    
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True: #making sure capture was succesful
                    frame_num += 1
                    time = time_ns()
                    data = [frame, time, frame_num]
                    s.sendall(pickle.dumps(data))
                else:
                    data = "Data not collected"
                    s.sendall(pickle.dumps(data))
        except socket.error:
            print("Video socket broken")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
        

if __name__ == '__main__':
    send_vid_data()
