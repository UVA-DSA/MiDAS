import tkinter as tk
from tkinter import *
import tkinter.ttk as ttk
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
    #Global vars
    global camOne,capOne, pUL, pUR, pLL, pLR, pClutch, pCam, pLong, camOneInd, camTwoInd, capOneInd, capTwoInd, trakInd, ardInd, watchLeftInd, watchRightInd, trak1Data, trak2Data,trak3Data,trak4Data,watchLData,watchRData
    # Window creation
    win.title("MIDAS V3 - Data Collection System")
    # set minimum window size value
    win.minsize(1080, 300)
    # set maximum window size value
    win.maxsize(1920, 1080)

    # Define a larger font
    large_font = ('Helvetica', 14)
    large_font_bold = ('Helvetica', 18, 'bold')

    notebook = ttk.Notebook(win)
    notebook.grid(row=0,column=1,rowspan=1,sticky='ne')

    # Info Frame
    infoFr = tk.LabelFrame(win, text="Information", padx=10, pady=10, font=large_font_bold)
    notebook.add(infoFr, text="Info")
    # Labels for the info frame
    tk.Label(infoFr, text="Subject:", font=large_font).grid(column=0, row=0, sticky="w")
    tk.Label(infoFr, text="Trial:", font=large_font).grid(column=2, row=0, sticky="w")
    tk.Label(infoFr, text="Task:", font=large_font).grid(column=0, row=2, sticky="w")
    tk.Label(infoFr, text="Rate:", font=large_font).grid(column=2, row=2, sticky="w")
    # Entry boxes for the info frame
    tk.Entry(infoFr, textvariable=subject_var, font=large_font, width=10).grid(column=1, row=0, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=trial_var, font=large_font, width=10).grid(column=3, row=0, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=task_var, font=large_font, width=10).grid(column=1, row=2, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=rate_var, font=large_font, width=10).grid(column=3, row=2, padx=5, pady=5, sticky="ew")
    # Submit Button
    tk.Button(infoFr, text='Submit', command=submitData, font=large_font, width=5).grid(column=4, row=0, rowspan=2, pady=10,padx=10, sticky='ns')

    # Pedal Indicators Frame
    #Makes the pedal frame and grids it
    pedalFr = tk.LabelFrame(win, text="Pedal Indicators", padx=10, pady=10, font=large_font_bold)
    pedalFr.grid(column=2, row=0, padx=10, pady=10, sticky="nsew", columnspan=1)
    #Makes vars for each pedal indicator
    pUL = tk.Label(pedalFr)
    pLL = tk.Label(pedalFr)
    pUR = tk.Label(pedalFr)
    pLR = tk.Label(pedalFr)
    pClutch = tk.Label(pedalFr)
    pCam = tk.Label(pedalFr)
    pLong = tk.Label(pedalFr)
    #List of pedals, their location, and correspodning var
    pedals = [
        ("U Left", 0, 2, 1, pUL), ("U Right", 0, 3, 1,pUR),
        ("L Left", 1, 2, 1, pLL), ("L Right", 1, 3, 1,pLR),
        ("Clutch", 0, 1, 1, pClutch), ("Camera", 1, 1, 1,pCam), ("Long", 0, 0, 2,pLong)
    ]
    #Takes each pedal var and configures it based on the list above
    for pedal in pedals:
        text, row, col, rowspan, label = pedal
        label.config(text=text, bg="red", width=8, height=3, font=large_font)
        label.grid(row=row, column=col, padx=5, pady=5, rowspan=rowspan)

    # Camera Frame
    #Creates Frame for the camera displays as well as the cam display itself and grids them
    camFr = tk.LabelFrame(win, text="Cameras", padx=10, pady=10, font=large_font_bold)
    camFr.grid(column=1,row=1, padx=10,pady=10,sticky='ne', rowspan=2, columnspan=2)
    camOne = tk.Label(camFr, width = 80, height = 15,bg="black")
    camTwo = tk.Label(camFr, width = 80, height = 15,bg="black")
    capOne =  tk.Label(camFr, width = 80, height = 15,bg="black")
    capTwo =  tk.Label(camFr, width = 80, height = 15,bg="black")
    camOne.grid(column=0,row=0,padx=10,pady=10,sticky="nsew")
    camTwo.grid(column=0,row=1,padx=10,pady=10,sticky="nsew")
    capOne.grid(column=1,row=0,padx=10,pady=10,sticky="nsew")
    capTwo.grid(column=1,row=1,padx=10,pady=10,sticky="nsew")

    # Data Frame
    #Creates the two frames which will hold data and grids them
    dataFr = tk.LabelFrame(win, text="Data", padx=15, pady=15, font=large_font_bold)
    dataFr.grid(column=0,row=0,padx=10,pady=10,sticky='nw', rowspan=3)
    watchFr = tk.LabelFrame(dataFr, text="Watches",padx=5,pady=5,font=large_font, width=30)
    watchFr.grid(column=0,row=0)
    trakFr = tk.LabelFrame(dataFr,text="Trakstar", padx=10,pady=10,font=large_font)
    trakFr.grid(column=0,row=1)
    #Creation of the data displays for each of the devices
    watchLData = tk.Label(watchFr)
    watchRData = tk.Label(watchFr)
    trak1Data = tk.Label(trakFr)
    trak2Data = tk.Label(trakFr)
    trak3Data = tk.Label(trakFr)
    trak4Data = tk.Label(trakFr)
    #List of the devices collecting data with their correspodning vars, positions, and frames
    dataDevices = [
        ("WatchL:", watchLData, 0, 0,watchFr), ("WatchR:", watchRData, 1, 0,watchFr),
        ("Trak1:", trak1Data, 0, 0,trakFr), ("Trak2:", trak2Data, 1, 0,trakFr),
        ("Trak3:", trak3Data, 2, 0,trakFr), ("Trak4:", trak4Data, 3, 0,trakFr)
    ]
    #Loops through and configures the data displays, creates labels for them, and grids them all in their corresponding frames
    for i, (label, name, r, c,frame) in enumerate(dataDevices):
        tk.Label(frame, text=label, font=large_font).grid(row=r, column=c, padx=5, pady=5, sticky="nw")
        name.config(font=large_font, text="X  X\nY  Y\nZ  Z")
        name.grid(row=r, column=c+1, padx=5, pady=5, sticky="w")

    # Diagonstic Frame
    # #Creates the frame on the main window
    diagnosticFr = tk.LabelFrame(win, text="Diagnostics", padx=10, pady=10, font=large_font_bold)
    notebook.add(diagnosticFr,text="Diagnostic")
    #Creates vars for the labels that will be updated with the status of each deivce
    capOneInd = tk.Label(diagnosticFr)
    capTwoInd = tk.Label(diagnosticFr)
    watchLeftInd = tk.Label(diagnosticFr)
    watchRightInd = tk.Label(diagnosticFr)
    ardInd = tk.Label(diagnosticFr)
    trakInd = tk.Label(diagnosticFr)
    camOneInd = tk.Label(diagnosticFr)
    camTwoInd = tk.Label(diagnosticFr)
    #List of devices with corresponding var
    devices = [
        ("Endo Left", capOneInd,0,0), ("Endo Right", capTwoInd,0,2),
        ("Watch Left:", watchLeftInd,0,4), ("Watch Right:", watchRightInd,1,0),
        ("PDS:", ardInd,1,2), ("trakSTAR:", trakInd,1,4),
        ("HandCam:", camOneInd,2,0), ("SurgicalCam:", camTwoInd,2,2)
    ]
    #Loops through and creates labels as well as the indicators and grids them in diagonsticFr
    for i, (label, name, r, c) in enumerate(devices):
        tk.Label(diagnosticFr, text=label, font=large_font).grid(row=r, column=c, padx=5, pady=5, sticky="w")
        name.config(bg="yellow", width=8, font=large_font, text="Not Found")
        name.grid(row=r, column=c+1, padx=5, pady=5, sticky="w")

    # Configure weight for responsiveness
    for i in range(2):
        win.grid_columnconfigure(i, weight=1)
    for i in range(3):
        win.grid_rowconfigure(i, weight=1)
    #Main process runs GUI, submit button triggers the new thread/processes
    win.mainloop()

#Function triggered by the submit button that forwards the infoFr fields into the main function in mp_app
def submitData():
    from mp_app import sendData
    sendData(subject_var.get(), trial_var.get(), task_var.get(), rate_var.get())

#Function that is run by a thread to update the GUI created above
def updateIndicators(gui_q,cam_q, OBS_q):
    while True:
        try:
            out = gui_q.get(True, 0.05)
            #Pedal Updator
            #-2 corresponds to no data from pedals in pulled data
            if(out[26] == -2):
                ardInd.config(text="No Data")
            else:
                #Loops through all of the pedal data
                for i in range(26,33):
                    ardInd.config(bg="green",text="Good")
                    temp = out[i]
                    if (temp>-1):
                        #Sets hexcolor to be a hex color from red to green depedning on how much pressure
                        red = int(255 * (1 - (temp - 1) / 1015))
                        green = int(255 * ((temp - 1) / 1015))
                        hex_color = f'#{red:02x}{green:02x}00'
                    else:
                        #If pedal data was -1 (no pressure) sets it to red
                        hex_color = "red"
                        temp = 0
                    #Displays the hexcolor and text to the corresponding pedal indicator determined by the letter/which data point the loop is on
                    pUL.config(bg = hex_color if i == 26 and temp != -2 else pUL.cget('bg'), text = "U Left \n" + str(temp) if i == 26 and temp != -2 else pUL.cget('text'))
                    pUR.config(bg = hex_color if i == 27 and temp != -2 else pUR.cget('bg'), text = "U Right \n" + str(temp) if i == 27 and temp != -2 else pUR.cget('text'))
                    pLL.config(bg = hex_color if i == 28 and temp != -2 else pLL.cget('bg'), text = "L Left \n" + str(temp) if i == 28 and temp != -2 else pLL.cget('text'))
                    pLR.config(bg = hex_color if i == 29 and temp != -2 else pLR.cget('bg'), text = "L Right \n" + str(temp) if i == 29 and temp != -2 else pLR.cget('text'))
                    pClutch.config(bg = hex_color if i == 30 and temp != -2 else pClutch.cget('bg'), text = "Clutch \n" + str(temp) if i == 30 and temp != -2 else pClutch.cget('text'))
                    pCam.config(bg = hex_color if i == 31 and temp != -2 else pCam.cget('bg'), text = "Camera \n" + str(temp) if i == 31 and temp != -2 else pCam.cget('text'))
                    pLong.config(bg = hex_color if i == 32 and temp != -2 else pLong.cget('bg'), text = "Long \n" + str(temp) if i == 32 and temp != -2 else pLong.cget('text'))
            #Trakstar Updator
            if(str(out[8]) == "0"):
                trakInd.config(text="No Data")
            else:
                trakInd.config(text="Good", bg="green")
                #Formats string of trakstar data to be displayed
                data = str(out[2]) + '  ' + str(out[5]) + '\n' + str(out[3]) + '  ' + str(out[6]) + '\n' + str(out[4]) + '  ' + str(out[7])
                print(data)
                #Updates the corresponding data field depedning on what sensor the data is from
                if(str(out[1]) == "0"):
                    trak1Data.config(text=data)
                elif(str(out[1]) == "1"):
                    trak2Data.config(text=data)
                elif(str(out[1]) == "2"):
                    trak3Data.config(text=data)
                elif(str(out[1]) == "3"):
                    trak4Data.config(text=data)
            #WatchL/R Updator
            if(out[14] == 0):
                watchLeftInd.config(text="No Data")
            else:
                watchLeftInd.config(text="Good", bg="green")
                data = str(round(out[17],4)) + '\n' + str(round(out[18],4)) + '\n' + str(round(out[19],4))
                watchLData.config(text=data)
            if(out[20]==0):
                watchRightInd.config(text="No Data")
            else:
                watchRightInd.config(text="Good", bg="green")
                data = str(round(out[23],4)) + '\n' + str(round(out[24],4)) + '\n' + str(round(out[25],4))
                watchRData.config(text=data)
         
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Exiting main program.")
            break
        #Sets indicators if queue has no data
        except queue.Empty:
            ardInd.config(bg="red", text="Error")
            watchLeftInd.config(bg="red", text="Error")
            watchRightInd.config(bg="red", text="Error")
            trakInd.config(bg="red", text="Error")
        #Camera Display Updator
        try:
            #Gets image and converts it from cv2 to pillow formats
            image = cam_q.get(False)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image.thumbnail((600,300))
            image = ImageTk.PhotoImage(image)
            #Displays image and sets indicator as good
            camOneInd.config(bg="green", text="Good")
            camOne.configure(image=image, height=300, width=600)
            camOne.image = image
        #Sets indicator is queue has no data
        except queue.Empty:
            camOneInd.config(bg="red",text="Error")
         #OBS Display Updator
        try:
            #Gets image and converts it from cv2 to pillow formats
            image = OBS_q.get(False)
            image = cv.resize(image,(600,300))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            #Displays image and sets indicator as good
            capOneInd.config(bg="green", text="Good")
            capOne.configure(image=image, height=300, width=600)
            capOne.image = image
        #Sets indicator is queue has no data
        except queue.Empty:
            capOneInd.config(bg="red",text="Error")

    win.destroy()