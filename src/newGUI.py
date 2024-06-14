import tkinter as tk

def startGui():
    print("test")

def getData():
    return [subject_var.get(), trial_var.get(),task_var.get(),rate_var.get()]

#Window and the 3 Frames creation
win = tk.Tk()
infoFr = tk.Frame(win, bd = 3, padx = 25, pady = 25)
pedalFr = tk.Frame(win, bd = 3, padx = 25, pady = 25)
diagnosticFr = tk.Frame(win, bd = 3, padx = 25, pady = 25)
infoFr.grid(column=0, row=0,sticky=tk.W)
pedalFr.grid(column=0,row=1)
diagnosticFr.grid(column=1,row=0,rowspan=2,sticky=tk.N)

#Vars for the info of each trial
subject_var = tk.StringVar()
trial_var = tk.StringVar()
task_var = tk.StringVar()
rate_var = tk.StringVar()
#Labels for the info frame
subject_label = tk.Label(infoFr, text="Subject:")
trial_label = tk.Label(infoFr, text="Trial:")
task_label = tk.Label(infoFr, text="Task:")
rate_label = tk.Label(infoFr, text="Rate:")
#Entry boxes for the info frame
subject_entry = tk.Entry(infoFr, textvariable=subject_var)
trial_entry = tk.Entry(infoFr, textvariable=trial_var)
task_entry = tk.Entry(infoFr, textvariable=task_var)
rate_entry = tk.Entry(infoFr, textvariable=rate_var)
sub_btn=tk.Button(infoFr, text = 'Submit',command=getData)
#Putting the labels and entries onto the Info Frame Grid
subject_label.grid(column=0,row=0)
trial_label.grid(column=0,row=1)
task_label.grid(column=0,row=2)
rate_label.grid(column=0,row=3)
subject_entry.grid(column=1,row=0)
trial_entry.grid(column=1,row=1)
task_entry.grid(column=1,row=2)
rate_entry.grid(column=1,row=3)
sub_btn.grid(column=0,row=4,columnspan=2)

#Pedal Indicators
pUL = tk.Label(pedalFr, text="Upper Left", bg="red",width=10,height=5)
pUR = tk.Label(pedalFr, text="Upper Right", bg="red",width=10,height=5)
pLL = tk.Label(pedalFr, text="Lower Left", bg="red",width=10,height=5)
pLR = tk.Label(pedalFr, text="Lower Right", bg="red",width=10,height=5)
pClutch = tk.Label(pedalFr, text="Clutch", bg="red",width=10,height=5)
pCam = tk.Label(pedalFr, text="Camera", bg="red",width=10,height=5)
pLong = tk.Label(pedalFr, text="Long", bg="red",width=10,height=10)
#Placing Pedal Indicators on Pedal Frame Grid
pClutch.grid(row=0, column=1,padx=10,pady=10)
pCam.grid(row=1, column=1,padx=10,pady=10)
pUL.grid(row=0, column=2,padx=20,pady=10)
pLL.grid(row=1, column=2,padx=10,pady=10)
pUR.grid(row=0, column=3,padx=10,pady=10)
pLR.grid(row=1, column=3,padx=10,pady=10)
pLong.grid(row=0, column=0,rowspan=3,padx=10,pady=10)

#Label creation for the device statuses
capOneLabel = tk.Label(diagnosticFr, text="Capture 1:")
capTwoLabel = tk.Label(diagnosticFr, text="Capture 2:")
watchLeftLabel = tk.Label(diagnosticFr, text="Watch Left:")
watchRightLabel = tk.Label(diagnosticFr, text="Watch Right:")
arduinoLabel = tk.Label(diagnosticFr, text="Arduino (Pedals):")
#Status boxes for the device statuses
capOneStatus = tk.Label(diagnosticFr, text="Error",bg="red")
capTwoStatus = tk.Label(diagnosticFr, text="Error",bg="red")
watchLeftStatus = tk.Label(diagnosticFr, text="Error",bg="red")
watchRightStatus = tk.Label(diagnosticFr, text="Error",bg="red")
arduinoStatus = tk.Label(diagnosticFr, text="Error",bg="red")
#Gridding labels and boxes onto diagnostic frame grid
capOneLabel.grid(row=0,column=0,padx=10,pady=10)
capTwoLabel.grid(row=1,column=0,padx=10,pady=10)
watchLeftLabel.grid(row=2,column=0,padx=10,pady=10)
watchRightLabel.grid(row=3,column=0,padx=10,pady=10)
arduinoLabel.grid(row=4,column=0,padx=10,pady=10)
capOneStatus.grid(row=0,column=1,padx=10,pady=10)
capTwoStatus.grid(row=1,column=1,padx=10,pady=10)
watchLeftStatus.grid(row=2,column=1,padx=10,pady=10)
watchRightStatus.grid(row=3,column=1,padx=10,pady=10)
arduinoStatus.grid(row=4,column=1,padx=10,pady=10)

tk.mainloop()