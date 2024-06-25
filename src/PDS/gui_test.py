import tkinter as tk
import time
import serial
import threading
import queue
from tkinter import *
from PIL import ImageTk,Image
import cv2 as cv
import numpy as np

try:
    arduino = serial.Serial("COM3", baudrate=9600, timeout=.1)
except:
    print("serial not found")
file = open(".\src\PDS\out.txt", "a")

window = tk.Tk()
stopSignal = False

pUL = tk.Label(text="Upper Left", bg="red",width=10,height=5)
pUR = tk.Label(text="Upper Right", bg="red",width=10,height=5)
pLL = tk.Label(text="Lower Left", bg="red",width=10,height=5)
pLR = tk.Label(text="Lower Right", bg="red",width=10,height=5)
pClutch = tk.Label(text="Clutch", bg="red",width=10,height=5)
pCam = tk.Label(text="Camera", bg="red",width=10,height=5)
pLong = tk.Label(text="Long", bg="red",width=10,height=10)

capOne = tk.Label()
capOne.grid(column=4,row=0,padx=10,pady=10,sticky="nsew", rowspan=2)
cap = cv.VideoCapture(0)

pClutch.grid(row=0, column=1,padx=10,pady=10)
pCam.grid(row=1, column=1,padx=10,pady=10)
pUL.grid(row=0, column=2,padx=20,pady=10)
pLL.grid(row=1, column=2,padx=10,pady=10)
pUR.grid(row=0, column=3,padx=10,pady=10)
pLR.grid(row=1, column=3,padx=10,pady=10)
pLong.grid(row=0, column=0,rowspan=3,padx=10,pady=10)

def serialReciever():
    threadCapture.start()
    while True:
        out = arduino.readline().decode()
        print(out, end = '')
        file.write(str(round((1000 * time.time()))) + "\t" + out.strip() +"\n")
        if '1' in out:
            pUL.config(bg="green")
        else:
            pUL.config(bg="red")
        if '2' in out:
            pUR.config(bg="green")
        else:
            pUR.config(bg="red")
        if '3' in out:
            pLL.config(bg="green")
        else:
            pLL.config(bg="red")
        if '4' in out:
            pLR.config(bg="green")
        else:
            pLR.config(bg="red")
        if '5' in out:
            pClutch.config(bg="green")
        else:
            pClutch.config(bg="red")
        if '6' in out:
            pCam.config(bg="green")
        else:
            pCam.config(bg="red")
        if '7' in out:
            pLong.config(bg="green")
        else:
            pLong.config(bg="red")



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

threadReciever = threading.Thread(target=serialReciever)
threadCapture = threading.Thread(target=video_playback)

start = tk.Button(text = "start", width = 20, height = 5, command = threadReciever.start)
start.grid(row = 3, column = 2, padx= 10, pady=10,columnspan=2)
threadLoop = threading.Thread(target=tk.mainloop())
threadLoop.start()
