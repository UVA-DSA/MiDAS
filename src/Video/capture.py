import cv2  #for capturing
import time
import socket
from time import time_ns
import pickle

def send_vid_data():
    cap = cv2.VideoCapture(1) #1 for connection to the video capture card
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 12345))
        
        num = 0


        
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True: #making sure capture was succesful
            time = time_ns()
            data = [frame, time]
            s.sendall(pickle.dumps(data))
        else:
            data = ["Data not collected", "N/A"]
            s.sendall(pickle.dumps(data))
    return

if __name__ == '__main__':
    send_vid_data()







#Actually starting video recording
print(Fore.YELLOW + "Video recording started")
i = 0


while cap.isOpened():
    ret, frame = cap.read()
    
    if i == 0:
        start_time_recording = round(time.time() * 1000)
        i+= 1
        
    
    if ret == True: #making sure capture was succesful
        out.write(frame)
        cv2.imshow("Output", frame)
        
        
        #Checking for when the kinematic data stops
        f = open("updateFile.txt", "r")
        data = f.read()
        f.close()
        if data == "end":
            os.remove("updateFile.txt")
            break
        
        #checking for errors in the kinematic data collection process
        g = open("errorFile.txt", "r")
        data = g.read()
        g.close()
        if data == "error":
            error_handler("Something went wrong in the kinematic data collection during recording", dir_name)
    
    else:
        error_handler("Unable to connect to video card", dir_name)

#shut down connection to capture card and errorFile (comm file not needed)
cap.release()
cv2.destroyAllWindows()
out.release()
os.remove("errorFile.txt")
print(Fore.YELLOW + "Video recording ended")

