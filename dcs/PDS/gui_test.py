import tkinter as tk
import time
import serial
import threading
import queue
from tkinter import *
from PIL import ImageTk,Image
import cv2 as cv
import numpy as np
import csv

#Make serial connection
try:
    arduino = serial.Serial("COM3", baudrate=9600, timeout=.1)
except:
    print("serial not found")

#Creation of csv writer
csv_path = 'src/PDS/PDS.csv'    #path of csv output
csv_file = open(csv_path, 'w', newline='')
csv_writer_PDS = csv.writer(csv_file)
csv_writer_PDS.writerow(['Pedals Pressed','Computer Time'])

#Root window
window = tk.Tk()
stopSignal = False

#Labels for pedals
pUL = tk.Label(text="Upper Left", bg="red",width=10,height=5)
pUR = tk.Label(text="Upper Right", bg="red",width=10,height=5)
pLL = tk.Label(text="Lower Left", bg="red",width=10,height=5)
pLR = tk.Label(text="Lower Right", bg="red",width=10,height=5)
pClutch = tk.Label(text="Clutch", bg="red",width=10,height=5)
pCam = tk.Label(text="Camera", bg="red",width=10,height=5)
pLong = tk.Label(text="Long", bg="red",width=10,height=10)

#Label to display camera capture
capOne = tk.Label()
capOne.grid(column=4,row=0,padx=10,pady=10,sticky="nsew", rowspan=2)
# cap = cv.VideoCapture(0)    #Capture object (0 is webcam)

#Places all the pedals on the root window grid
pClutch.grid(row=0, column=1,padx=10,pady=10)
pCam.grid(row=1, column=1,padx=10,pady=10)
pUL.grid(row=0, column=2,padx=20,pady=10)
pLL.grid(row=1, column=2,padx=10,pady=10)
pUR.grid(row=0, column=3,padx=10,pady=10)
pLR.grid(row=1, column=3,padx=10,pady=10)
pLong.grid(row=0, column=0,rowspan=3,padx=10,pady=10)

#Each letter corresopnds to a pedal output. Arduino will output pedal data as single string
#format if 'A123' meaning the first pedal was pressed at pressure of 123. 'F999 G200' means last two pedals pressed, second to last being pressed harder
letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G'}
def serialReciever():
    #threadCapture.start()
    while True:
        out = arduino.readline().decode()
        print(out)  
        # csv_writer_PDS.writerow([out.strip(), time.time_ns()])
        # csv_file.flush()
        #Cycles through each potienital letter and sees if present in output
        for let in letters:
            if (out.find(let) != -1):
                temp = float((out[(1 + out.find(let)):(5 + out.find(let))]).strip())
                #If it is, sets a color from red to green depedning on how much pressure
                red = int(255 * (1 - (temp - 1) / 11000))
                green = int(255 * ((temp - 1) / 11000))
                if(temp > 2000):
                    hex_color = 'green'
                else:
                    hex_color = f'#{red:02x}{green:02x}00'

            else:
                hex_color = "red"
                temp = 0
            #Sets the color of corresponding pedal on GUI
            pUL.config(bg = hex_color if let == 'A' else pUL.cget('bg'), text = "Upper Left \n" + str(temp) if let == 'A' else pUL.cget('text'))
            pUR.config(bg = hex_color if let == 'B' else pUR.cget('bg'), text = "Upper Right \n" + str(temp) if let == 'B' else pUR.cget('text'))
            pLL.config(bg = hex_color if let == 'C' else pLL.cget('bg'), text = "Lower Left \n" + str(temp) if let == 'C' else pLL.cget('text'))
            pLR.config(bg = hex_color if let == 'D' else pLR.cget('bg'), text = "Lower Right \n" + str(temp) if let == 'D' else pLR.cget('text'))
            pClutch.config(bg = hex_color if let == 'E' else pClutch.cget('bg'), text = "Clutch \n" + str(temp) if let == 'E' else pClutch.cget('text'))
            pCam.config(bg = hex_color if let == 'F' else pCam.cget('bg'), text = "Camera \n" + str(temp) if let == 'F' else pCam.cget('text'))
            pLong.config(bg = hex_color if let == 'G' else pLong.cget('bg'), text = "Long\n" + str(temp) if let == 'G' else pLong.cget('text'))
            

def video_playback():
   #checks if file/camera is opened
   if not cap.isOpened():
      print("Cannot open File / Camera")
      return
   
   while cap.isOpened():
   #returns boolean if read correctly and image with the frame
      ret, frame = cap.read()
      # if frame is read correctly ret is True
      if not ret:
         print("Error in reading")
         continue
      #Frame conversion from cv2 to pillow so it can be displayed on the tkinter GUI
      frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
      frame = Image.fromarray(frame)
      frame = ImageTk.PhotoImage(frame)
      capOne.configure(image=frame)
      capOne.image = frame
      #if 'q' on keyboard is pressed, will end playback
      if cv.waitKey(1) == ord('q'):
         print("Q Pressed, Ending Playback Early")
         break
   cap.release()
   cv.destroyAllWindows()

#Creation of separate threads for video capture  and serial communication
threadReciever = threading.Thread(target=serialReciever)
threadCapture = threading.Thread(target=video_playback)

#Creation of button to trigger the two threads
start = tk.Button(text = "start", width = 20, height = 5, command = threadReciever.start)
start.grid(row = 3, column = 2, padx= 10, pady=10,columnspan=2)
tk.mainloop()
