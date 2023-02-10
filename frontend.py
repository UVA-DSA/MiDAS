from gettext import translation
from re import sub
import json 
import os
from datetime import datetime
from tkinter import *
from msilib.schema import RadioButton
from tkinter import messagebox as m_box
from cv2 import checkRange

root = Tk()

root.title("System Parameters")
root.geometry("450x750")

subject = ""
medOrRes = ""
year = ""
attendingPhysician = ""
task = ""
trial = ""
rate = 90.0
range = 36.0
metric = 1 # RADIO
angle_align = [0,0,0]
azim = 0
elev = 0
roll = 0
reference_frame = [0,0,0]
xyz_reference_frame = 1 # RADIO
hemisphere = 0 # RADIO
filter_ac_wide_notch = 1 # RADIO
filter_ac_narrow_notch = 0 # RADIO

subjectLabel = Label(root, text = "Enter Subject", padx = 10, pady = 10,  bg = 'pink', fg = 'black')
subjectLabel.grid(row = 0, column = 1)

subjectEntry = Entry(root, width = 25, fg = 'black')
subjectEntry.grid(row = 1, column = 1)

medOrResLabel = Label(root, text = "Medical Student or Resident", padx = 10, pady = 10, bg = 'pink', fg = 'black')
medOrResLabel.grid(row = 2, column = 1)

medOrResEntry = StringVar()
medOrResEntry.set("N/A")

Radiobutton(root, text = "Medical Student", font = ("Times New Roman", 10, 'bold'), variable = medOrResEntry, value = "Medical Student").grid(row = 3, column = 1)
Radiobutton(root, text = "Resident", font = ("Times New Roman", 10, 'bold'), variable = medOrResEntry, value = "Resident").grid(row = 3, column = 2)
Radiobutton(root, text = "N/A", font = ("Times New Roman", 10, 'bold'), variable = medOrResEntry, value = "N/A").grid(row = 4, column = 1)

yearLabel = Label(root, text = "Enter Year", padx = 10, pady = 10, bg = 'pink', fg = 'black')
yearLabel.grid(row = 5, column = 1)

yearEntry = StringVar()
yearEntry.set("N/A")

Radiobutton(root, text = "1", font = ("Times New Roman", 10, 'bold'), variable = yearEntry, value = "1").grid(row = 6, column = 1)
Radiobutton(root, text = "2", font = ("Times New Roman", 10, 'bold'), variable = yearEntry, value = "2").grid(row = 6, column = 2)
Radiobutton(root, text = "3", font = ("Times New Roman", 10, 'bold'), variable = yearEntry, value = "3").grid(row = 7, column = 1)
Radiobutton(root, text = "4", font = ("Times New Roman", 10, 'bold'), variable = yearEntry, value = "4").grid(row = 7, column = 2)
Radiobutton(root, text = "N/A", font = ("Times New Roman", 10, 'bold'), variable = yearEntry, value = "N/A").grid(row = 8, column = 1)

attendingPhysicianLabel = Label(root, text = "Enter Attending Physician", padx = 10, pady = 10, bg = 'pink', fg = 'black')
attendingPhysicianLabel.grid(row = 9, column = 1)

attendingPhysicianEntry = Entry(root, width = 25, fg = 'black')
attendingPhysicianEntry.grid(row = 10, column = 1)

taskLabel = Label(root, text = "Enter Task", padx = 10, pady = 10, bg = 'pink', fg = 'black')
taskLabel.grid(row = 11, column = 1)

taskEntry = Entry(root, width = 25, fg = 'black')
taskEntry.grid(row = 12, column = 1)

trialLabel = Label(root, text = "Enter Trial", padx = 10, pady = 10, bg = 'pink', fg = 'black')
trialLabel.grid(row = 11, column = 2)

trialEntry = Entry(root, width = 25, fg = 'black')
trialEntry.grid(row = 12, column = 2)

button1 = Label(root, text = "Enter Rate", padx = 10, pady = 10, bg = 'pink', fg = 'black')
button1.grid(row = 13, column = 1)

rateEntry = Entry(root, width = 25, fg = 'black')
rateEntry.insert(0, "90.0")
rateEntry.grid(row = 14, column = 1)

button2 = Label(root, text = "Enter Range", padx = 10, pady = 10, bg = 'pink', fg = 'black')
button2.grid(row = 13, column = 2)

rangeEntry = DoubleVar()
rangeEntry.set(36.0)

Radiobutton(root, text = "36.0 in", font = ("Times New Roman", 10, 'bold'), variable = rangeEntry, value = 36.0).grid(row = 14, column = 2)
Radiobutton(root, text = "72.0 in", font = ("Times New Roman", 10, 'bold'), variable = rangeEntry, value = 72.0).grid(row = 15,  column = 2)
Radiobutton(root, text = "144.0 in", font = ("Times New Roman", 10, 'bold'), variable = rangeEntry, value = 144.0).grid(row = 16,  column = 2)

button8 = Label(root, text = "Enter Hemisphere", padx = 10, pady = 10, bg = 'pink', fg = 'black')
button8.grid(row = 17, column = 1)

hemisphereEntry = IntVar()
hemisphereEntry.set(0)

Radiobutton(root, text = "Front", font = ("Times New Roman", 10, 'bold'), variable = hemisphereEntry, value = 0).grid(row = 18, column = 1)
Radiobutton(root, text = "Back", font = ("Times New Roman", 10, 'bold'), variable = hemisphereEntry, value = 1).grid(row = 18, column = 2)
Radiobutton(root, text = "Top", font = ("Times New Roman", 10, 'bold'), variable = hemisphereEntry, value = 2).grid(row = 19, column = 1)
Radiobutton(root, text = "Bottom", font = ("Times New Roman", 10, 'bold'), variable = hemisphereEntry, value = 3).grid(row = 19, column = 2)
Radiobutton(root, text = "Left", font = ("Times New Roman", 10, 'bold'), variable = hemisphereEntry, value = 4).grid(row = 20, column = 1)
Radiobutton(root, text = "Right", font = ("Times New Roman", 10, 'bold'), variable = hemisphereEntry, value = 5).grid(row = 20, column = 2)

button9 = Label(root, text = "Enter Filter", padx = 10, pady = 10, bg = 'pink', fg = 'black')
button9.grid(row = 21, column = 1)

filterAcNotchEntry = IntVar()
filterAcNotchEntry.set(1)

Radiobutton(root, text = "Wide", font = ("Times New Roman", 10, 'bold'), variable = filterAcNotchEntry, value = 1).grid(row = 22, column = 1)
Radiobutton(root, text = "Narrow", font = ("Times New Roman", 10, 'bold'), variable = filterAcNotchEntry, value = 0).grid(row = 22,  column = 2)

def masterFunc():
    checker = True
    subject = subjectEntry.get()
    medOrRes = medOrResEntry.get()
    year = yearEntry.get()
    attendingPhysician = attendingPhysicianEntry.get()
    task = taskEntry.get()
    trial = trialEntry.get()
    
    if (subject=="" or attendingPhysician=="" or task=="" or trial==""):
        m_box.showwarning("Error", "Please fill in the required fields.")
        checker = False
    
    rate = rateEntry.get()
    
    if (rate==""):
        m_box.showwarning("Error", "Please fill in the rate.")
        checker = False
    else:
        try:
            rate = float(rate)
        except ValueError:
            m_box.showerror("Title", "Please type in a number.")
            checker = False
    
    if not (255 > rate > 12):
        m_box.showerror("Title", "Please type in a number within the range of 12 and 255.")
        checker = False
    
    if checker:
        rate = float(rate)
        range = float(rangeEntry.get())
        hemisphere = hemisphereEntry.get()
        if (filterAcNotchEntry.get()==1):
            filter_ac_wide_notch = 1
            filter_ac_narrow_notch = 0
        else:
            filter_ac_wide_notch = 0
            filter_ac_narrow_notch = 1
            
        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
        dir_name = str(task) + "_S" + str(subject) + "_T" + str(trial) + "_" +  str(date_time)
        
        write_to_json_data = {
        "subject" :  subject,                               #
        "medOrRes": medOrRes,                               #
        "year": year,                                        #
        "attendingPhysician": attendingPhysician,           #
        "task": task,                                       #
        "trial": trial,                                     #
        "rate": rate,                                       #
        "range": range,                                     #
        "azimuth": azim,                                    #  
        "elevation":  elev,                                 #
        "roll": roll,                                       #
        "xyz_reference_frame": xyz_reference_frame,         #
        "hemisphere": hemisphere,                           #
        "filter_ac_wide_notch": filter_ac_wide_notch,       #
        "filter_ac_narrow_notch":  filter_ac_narrow_notch,  #
        "date_time": date_time,                             #
        "file_naming": dir_name,                            #
        }

        json_object = json.dumps(write_to_json_data)
        
        with open("data.json", 'w') as outfile:
            outfile.write(json_object)
        
        root.destroy()

        os.system("python capture.py")
        
        

finalButton = Button(root, text = "Start Data Collection", padx = 20, pady = 20, bg = 'black', fg = 'pink', command = lambda: masterFunc())
finalButton.grid(row = 23, column = 1, columnspan=2)

# legend = Label(root, text = "Rate - Measurement rate in Hz. Must be a decimal value within the range of 12 and 255.\nRange - Maximum range in any of the 3 axes in inches.\nAc Wide Notch Filter - Eliminates noise between 30 and 72 Hz but has slower computation time.\nAc Narrow Notch Filter - Eliminates narrower band of noise but has faster computation time.\n*Azimuth, elevation, and roll are sensor specific.", padx = 40, pady = 10, bg = 'white', fg = 'black', justify = 'left')
# legend.grid(row = 24, column = 1, columnspan = 2)

Font_tuple = ("Times New Roman", 10, 'bold')

subjectEntry.configure(font = Font_tuple)
subjectLabel.configure(font = Font_tuple)
medOrResLabel.configure(font = Font_tuple)
yearLabel.configure(font = Font_tuple)
attendingPhysicianLabel.configure(font = Font_tuple)
attendingPhysicianEntry.configure(font = Font_tuple)
taskEntry.configure(font = Font_tuple)
taskLabel.configure(font = Font_tuple)
trialEntry.configure(font = Font_tuple)
trialLabel.configure(font = Font_tuple)
button1.configure(font = Font_tuple)
rateEntry.configure(font = Font_tuple)
button2.configure(font = Font_tuple)
button8.configure(font = Font_tuple)
button9.configure(font = Font_tuple)
finalButton.configure(font = Font_tuple)
# legend.configure(font = Font_tuple)

root.mainloop()