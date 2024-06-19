import tkinter as tk
from tkinter import font
from tkinter import *
from PIL import ImageTk,Image
import queue

win = tk.Tk()
# Vars for the info of each trial
subject_var = tk.StringVar()
trial_var = tk.StringVar()
task_var = tk.StringVar()
rate_var = tk.StringVar()

def startGui(capture_q, trackstar_q, PDS_q):
    global pUL, pUR, pLL, pLR, pCam, pClutch, pLong
    global capOneInd, capTwoInd, camOneInd, camTwoInd, arduinoInd, trackstarInd, watchLeftInd, watchRightInd
    # Window and the 3 Frames creation
    win.title("MIDAS V3 - Data Collection System")
    # win.geometry("1800x1200")
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
    pUR = tk.Label(pedalFr)
    pLL = tk.Label(pedalFr)
    pLR = tk.Label(pedalFr)
    pCam= tk.Label(pedalFr)
    pClutch = tk.Label(pedalFr)
    pLong = tk.Label(pedalFr)

    pedals = [
        ("Upper Left", 0, 2, 1, pUL), ("Upper Right", 0, 3, 1, pUR),
        ("Lower Left", 1, 2, 1, pLL), ("Lower Right", 1, 3, 1, pLR),
        ("Clutch", 0, 1, 1, pClutch), ("Camera", 1, 1, 1, pCam), ("Long", 0, 0, 3, pLong)
    ]

    for pedal in pedals:
        text, row, col, rowspan, pedalName = pedal
        pedalName.config(text=text, bg="red", width=9, height=4, font=large_font)
        pedalName.grid(row=row, column=col, padx=10, pady=10, rowspan=rowspan)

    # Capture Frame
    captureFr = tk.LabelFrame(win, text="Captures", padx=15, pady=15, font=large_font_bold)
    captureFr.grid(column=2, row=0, rowspan=2, padx=10, pady=10, sticky="nsew")

    capOne = tk.Canvas(captureFr, width = 500, height = 300)
    capOne.grid(column=0,row=0,padx=10,pady=10,sticky="nsew")
    image = ImageTk.PhotoImage(Image.open("test/test.jpg"))
    capOne.create_image(108,72, anchor="center", image = image)

    # Diagonstic Frame
    global diagnosticFr
    diagnosticFr = tk.LabelFrame(win, text="Diagnostics", padx=15, pady=15, font=large_font_bold)
    diagnosticFr.grid(column=1, row=0, rowspan=1, padx=10, pady=10, sticky="nsew")
    capOneInd = tk.Label(diagnosticFr)
    capTwoInd = tk.Label(diagnosticFr)
    watchLeftInd = tk.Label(diagnosticFr)
    watchRightInd = tk.Label(diagnosticFr)
    arduinoInd = tk.Label(diagnosticFr)
    trackstarInd = tk.Label(diagnosticFr)
    camOneInd = tk.Label(diagnosticFr)
    camTwoInd = tk.Label(diagnosticFr)
    devices = [
        ("Capture 1:", capOneInd), ("Capture 2:",  capTwoInd),
        ("Watch Left:", watchLeftInd), ("Watch Right:",  watchRightInd),
        ("Arduino (Pedals):", arduinoInd), ("Trackstar:",  trackstarInd),
        ("Camera 1:",  camOneInd), ("Camera 2:", camTwoInd)
    ]   

    for i, (label, name) in enumerate(devices):
        tk.Label(diagnosticFr, text=label, font=large_font).grid(row=i, column=0, padx=10, pady=5, sticky="w")
        name.config(bg="yellow", width=8, font=large_font)
        name.grid(row=i, column=1, padx=10, pady=5, sticky="w")
    
    # Configure weight for responsiveness
    for i in range(2):
        win.grid_columnconfigure(i, weight=1)
    for i in range(3):
        win.grid_rowconfigure(i, weight=1)

    #diagonosticUpdates(capture_q, trackstar_q, PDS_q)
    pedalUpdates(PDS_q)
    win.mainloop()

def submitData():
    from app import sendData
    sendData(subject_var.get(), trial_var.get(), task_var.get(), rate_var.get())

def diagonosticUpdates(capture_q, trackstar_q, PDS_q):
    if capture_q.empty() == True:
        capOneInd.config(bg = "red", text="Error")
        capTwoInd.config(bg = "red", text="Error")
    else:
        capOneInd.config(bg = "green", text="Good")
        capTwoInd.config(bg = "red", text="Error")
    if trackstar_q.empty() == True:
        trackstarInd.config(bg = "red", text="Error")
    else:
        trackstarInd.config(bg = "green", text="Good")
    diagnosticFr.after(1000, diagonosticUpdates, capture_q,trackstar_q,PDS_q)

def pedalUpdates(PDS_q):
    try:
        out = PDS_q.get(True, 0.5)
        arduinoInd.config(bg = "green", text="Good")
    except queue.Empty:
        arduinoInd.config(bg = "red", text="Error")
    else:
        print(out)
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
    finally:
        pedalFr.after(1000,pedalUpdates,PDS_q)

