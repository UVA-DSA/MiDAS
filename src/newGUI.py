import tkinter as tk
from tkinter import font
from tkinter import *
from PIL import ImageTk,Image
import queue
import cv2 as cv

win = tk.Tk()
# Vars for the info of each trial
subject_var = tk.StringVar()
trial_var = tk.StringVar()
task_var = tk.StringVar()
rate_var = tk.StringVar()

def startGui():
    global capOne, pUL, pUR, pLL, pLR, pClutch, pCam, pLong, camOneInd, camTwoInd, capOneInd, capTwoInd, trakInd, ardInd, watchLeftInd, watchRightInd
    # Window creation
    win.title("MIDAS V3 - Data Collection System")
    # set minimum window size value
    win.minsize(1080, 300)
    # set maximum window size value
    win.maxsize(1920, 1080)

    # Define a larger font
    large_font = ('Helvetica', 16)
    large_font_bold = ('Helvetica', 20, 'bold')

    # Info Frame
    infoFr = tk.LabelFrame(win, text="Information", padx=15, pady=15, font=large_font_bold)
    infoFr.grid(column=0, row=0, padx=10, pady=10, sticky="nsew")

    # Labels for the info frame
    tk.Label(infoFr, text="Subject:", font=large_font).grid(column=0, row=0, sticky="w")
    tk.Label(infoFr, text="Trial:", font=large_font).grid(column=0, row=1, sticky="w")
    tk.Label(infoFr, text="Task:", font=large_font).grid(column=0, row=2, sticky="w")
    tk.Label(infoFr, text="Rate:", font=large_font).grid(column=0, row=3, sticky="w")

    # Entry boxes for the info frame
    tk.Entry(infoFr, textvariable=subject_var, font=large_font, width=15).grid(column=1, row=0, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=trial_var, font=large_font, width=15).grid(column=1, row=1, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=task_var, font=large_font, width=15).grid(column=1, row=2, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=rate_var, font=large_font, width=15).grid(column=1, row=3, padx=5, pady=5, sticky="ew")

    # Submit Button
    tk.Button(infoFr, text='Submit', command=submitData, font=large_font, width=5).grid(column=0, row=4, columnspan=2, pady=10)

    # Pedal Indicators Frame
    global pedalFr
    pedalFr = tk.LabelFrame(win, text="Pedal Indicators", padx=15, pady=15, font=large_font_bold)
    pedalFr.grid(column=0, row=1, padx=10, pady=10, sticky="nsew", columnspan=2)
    pUL = tk.Label(pedalFr)
    pLL = tk.Label(pedalFr)
    pUR = tk.Label(pedalFr)
    pLR = tk.Label(pedalFr)
    pClutch = tk.Label(pedalFr)
    pCam = tk.Label(pedalFr)
    pLong = tk.Label(pedalFr)
    pedals = [
        ("Upper Left", 0, 2, 1, pUL), ("Upper Right", 0, 3, 1,pUR),
        ("Lower Left", 1, 2, 1, pLL), ("Lower Right", 1, 3, 1,pLR),
        ("Clutch", 0, 1, 1, pClutch), ("Camera", 1, 1, 1,pCam), ("Long", 0, 0, 3,pLong)
    ]

    for pedal in pedals:
        text, row, col, rowspan, label = pedal
        label.config(text=text, bg="red", width=9, height=4, font=large_font)
        label.grid(row=row, column=col, padx=10, pady=10, rowspan=rowspan)

    # Capture Frame
    captureFr = tk.LabelFrame(win, text="Captures", padx=15, pady=15, font=large_font_bold)
    captureFr.grid(column=2, row=0, rowspan=2, padx=10, pady=10, sticky="nsew")

    capOne = tk.Label(captureFr, width = 80, height = 30,bg="black")
    capOne.grid(column=0,row=0,padx=10,pady=10,sticky="nsew")

    # Diagonstic Frame
    global diagnosticFr
    diagnosticFr = tk.LabelFrame(win, text="Diagnostics", padx=15, pady=15, font=large_font_bold)
    diagnosticFr.grid(column=1, row=0, rowspan=1, padx=10, pady=10, sticky="nsew")
    capOneInd = tk.Label(diagnosticFr)
    capTwoInd = tk.Label(diagnosticFr)
    watchLeftInd = tk.Label(diagnosticFr)
    watchRightInd = tk.Label(diagnosticFr)
    ardInd = tk.Label(diagnosticFr)
    trakInd = tk.Label(diagnosticFr)
    camOneInd = tk.Label(diagnosticFr)
    camTwoInd = tk.Label(diagnosticFr)

    devices = [
        ("Capture 1:", capOneInd), ("Capture 2:", capTwoInd),
        ("Watch Left:", watchLeftInd), ("Watch Right:", watchRightInd),
        ("Arduino (Pedals):", ardInd), ("Trackstar:", trakInd),
        ("Camera 1:", camOneInd), ("Camera 2:", camTwoInd)
    ]

    for i, (label, name) in enumerate(devices):
        tk.Label(diagnosticFr, text=label, font=large_font).grid(row=i, column=0, padx=10, pady=5, sticky="w")
        name.config(bg="yellow", width=8, font=large_font, text="Not Found")
        name.grid(row=i, column=1, padx=10, pady=5, sticky="w")

    # Configure weight for responsiveness
    for i in range(2):
        win.grid_columnconfigure(i, weight=1)
    for i in range(3):
        win.grid_rowconfigure(i, weight=1)

    win.mainloop()

def submitData():
    from app import sendData
    sendData(subject_var.get(), trial_var.get(), task_var.get(), rate_var.get())

def updateIndicators(PDS_q, capture_q):
    while True:
        try:
            out = PDS_q.get(True,0.1)
            ardInd.config(bg="green",text="Good")
            pUL.config(bg = "green" if '1' in out else "red")
            pUR.config(bg = "green" if '2' in out else "red")
            pLL.config(bg = "green" if '3' in out else "red")
            pLR.config(bg = "green" if '4' in out else "red")
            pClutch.config(bg = "green" if '5' in out else "red")
            pCam.config(bg = "green" if '6' in out else "red")
            pLong.config(bg = "green" if '7' in out else "red")
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Exiting main program.")
            break
        except queue.Empty:
            ardInd.config(bg="red", text="Error")
        try:
            image = capture_q.get(True,0.1)
            capOneInd.config(bg="green", text="Good")
            capOne.configure(image=image, height=300, width=500)
            capOne.image = image
        except queue.Empty:
            capOneInd.config(bg="red",text="Error")
    win.destroy()