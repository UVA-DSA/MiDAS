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
    global camOne, pUL, pUR, pLL, pLR, pClutch, pCam, pLong, camOneInd, camTwoInd, capOneInd, capTwoInd, trakInd, ardInd, watchLeftInd, watchRightInd, trak1Data, trak2Data,trak3Data,trak4Data,watchLData,watchRData
    # Window creation
    win.title("MIDAS V3 - Data Collection System")
    # set minimum window size value
    win.minsize(1080, 300)
    # set maximum window size value
    win.maxsize(1920, 1080)

    # Define a larger font
    large_font = ('Helvetica', 14)
    large_font_bold = ('Helvetica', 18, 'bold')

    # Info Frame
    infoFr = tk.LabelFrame(win, text="Information", padx=10, pady=10, font=large_font_bold)
    infoFr.grid(column=0, row=0, padx=10, pady=10, sticky="nsew")
    # Labels for the info frame
    tk.Label(infoFr, text="Subject:", font=large_font).grid(column=0, row=0, sticky="w")
    tk.Label(infoFr, text="Trial:", font=large_font).grid(column=0, row=1, sticky="w")
    tk.Label(infoFr, text="Task:", font=large_font).grid(column=0, row=2, sticky="w")
    tk.Label(infoFr, text="Rate:", font=large_font).grid(column=0, row=3, sticky="w")
    # Entry boxes for the info frame
    tk.Entry(infoFr, textvariable=subject_var, font=large_font, width=10).grid(column=1, row=0, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=trial_var, font=large_font, width=10).grid(column=1, row=1, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=task_var, font=large_font, width=10).grid(column=1, row=2, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=rate_var, font=large_font, width=10).grid(column=1, row=3, padx=5, pady=5, sticky="ew")
    # Submit Button
    tk.Button(infoFr, text='Submit', command=submitData, font=large_font, width=5).grid(column=0, row=4, columnspan=2, pady=10)

    # Pedal Indicators Frame
    #Makes the pedal frame and grids it
    pedalFr = tk.LabelFrame(win, text="Pedal Indicators", padx=10, pady=10, font=large_font_bold)
    pedalFr.grid(column=0, row=1, padx=10, pady=10, sticky="nsew", columnspan=2)
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
    camFr = tk.LabelFrame(win, text="Camera", padx=10, pady=10, font=large_font_bold)
    camFr.grid(column=2,row=0, padx=10,pady=10,sticky='nw')
    camOne = tk.Label(camFr, width = 80, height = 20,bg="black")
    camOne.grid(column=0,row=0,padx=10,pady=10,sticky="nsew")

    # Data Frame
    #Creates the two frames which will hold data and grids them
    dataFr = tk.LabelFrame(win, text="Data", padx=15, pady=15, font=large_font_bold)
    dataFr.grid(column=2,row=1,padx=10,pady=10,sticky='nw')
    watchFr = tk.LabelFrame(dataFr, text="Watches",padx=10,pady=10,font=large_font)
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
        ("WatchL:", watchLData, 0, 0,watchFr), ("WatchR:", watchRData, 0, 2,watchFr),
        ("Trak1:", trak1Data, 0, 0,trakFr), ("Trak2:", trak2Data, 0, 2,trakFr),
        ("Trak3:", trak3Data, 0, 4,trakFr), ("Trak4:", trak4Data, 0, 6,trakFr)
    ]
    #Loops through and configures the data displays, creates labels for them, and grids them all in their corresponding frames
    for i, (label, name, r, c,frame) in enumerate(dataDevices):
        tk.Label(frame, text=label, font=large_font).grid(row=r, column=c, padx=5, pady=5, sticky="nw")
        name.config(font=large_font, text="X  X\nY  Y\nZ  Z")
        name.grid(row=r, column=c+1, padx=5, pady=5, sticky="w")

    # Diagonstic Frame
    # #Creates the frame on the main window
    diagnosticFr = tk.LabelFrame(win, text="Diagnostics", padx=10, pady=10, font=large_font_bold)
    diagnosticFr.grid(column=1, row=0, rowspan=1, padx=10, pady=10, sticky="nsew")
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
        ("Capture 1:", capOneInd), ("Capture 2:", capTwoInd),
        ("Watch Left:", watchLeftInd), ("Watch Right:", watchRightInd),
        ("Arduino:", ardInd), ("Trackstar:", trakInd),
        ("Camera 1:", camOneInd), ("Camera 2:", camTwoInd)
    ]
    #Loops through and creates labels as well as the indicators and grids them in diagonsticFr
    for i, (label, name) in enumerate(devices):
        tk.Label(diagnosticFr, text=label, font=large_font).grid(row=i, column=0, padx=5, pady=5, sticky="w")
        name.config(bg="yellow", width=8, font=large_font, text="Not Found")
        name.grid(row=i, column=1, padx=5, pady=5, sticky="w")

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
def updateIndicators(gui_q,cam_q):
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
            if(out[7] == 0):
                trakInd.config(text="No Data")
            else:
                #Formats string of trakstar data to be displayed
                data = str(out[1]) + '  ' + str(out[4]) + '\n' + str(out[2]) + '  ' + str(out[5]) + '\n' + str(out[3]) + '  ' + str(out[6])
                #Updates the corresponding data field depedning on what sensor the data is from
                if(out[0] == 0):
                    trak1Data.config(text= data)
                elif(out[0] == 1):
                    trak2Data.config(text=data)
                elif(out[0] == 2):
                    trak3Data.config(text=data)
                elif(out[0] == 3):
                    trak4Data.config(text=data)
            #WatchL/R Updator
            if(out[14] == 0):
                watchLeftInd.config(text="No Data")
            else:
                data = str(out[16]) + '\n' + str(out[17]) + '\n' + str(out[19])
                watchLData.config(text=data)
            if(out[20]==0):
                watchRightInd.config(text="No Data")
            else:
                data = str(out[22]) + '\n' + str(out[23]) + '\n' + str(out[24])
         
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
            image = cam_q.get(True,0.05)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            #Displays image and sets indicator as good
            camOneInd.config(bg="green", text="Good")
            camOne.configure(image=image, height=300, width=600)
            camOne.image = image
        #Sets indicator is queue has no data
        except queue.Empty:
            camOneInd.config(bg="red",text="Error")

    win.destroy()