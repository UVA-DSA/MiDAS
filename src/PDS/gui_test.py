import tkinter as tk
import time
import serial
import threading

arduino = serial.Serial("COM3", baudrate=9600, timeout=.1)
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

pClutch.grid(row=0, column=1,padx=10,pady=10)
pCam.grid(row=1, column=1,padx=10,pady=10)
pUL.grid(row=0, column=2,padx=20,pady=10)
pLL.grid(row=1, column=2,padx=10,pady=10)
pUR.grid(row=0, column=3,padx=10,pady=10)
pLR.grid(row=1, column=3,padx=10,pady=10)
pLong.grid(row=0, column=0,rowspan=3,padx=10,pady=10)

def serialReciever():
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

threadReciever = threading.Thread(target=serialReciever)

start = tk.Button(text = "start", width = 20, height = 5, command = threadReciever.start)
start.grid(row = 3, column = 2, padx= 10, pady=10,columnspan=2)
threadLoop = threading.Thread(target=tk.mainloop())
threadLoop.start()
