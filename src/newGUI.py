import tkinter as tk
from tkinter import font

win = tk.Tk()
# Vars for the info of each trial
subject_var = tk.StringVar()
trial_var = tk.StringVar()
task_var = tk.StringVar()
rate_var = tk.StringVar()

def startGui():
    # Window and the 3 Frames creation
    win.title("MIDAS V3 - Data Collection System")
    # win.geometry("1800x1200")
    # set minimum window size value
    win.minsize(1080, 300)
    
    # set maximum window size value
    win.maxsize(1200, 900)

    # Define a larger font
    large_font = ('Helvetica', 18)
    large_font_bold = ('Helvetica', 24, 'bold')

    # Info Frame
    infoFr = tk.LabelFrame(win, text="Information", padx=15, pady=15, font=large_font_bold)
    infoFr.grid(column=0, row=0, padx=10, pady=10, sticky="nsew")

    # Labels for the info frame
    tk.Label(infoFr, text="Subject:", font=large_font).grid(column=0, row=0, sticky="w")
    tk.Label(infoFr, text="Trial:", font=large_font).grid(column=0, row=1, sticky="w")
    tk.Label(infoFr, text="Task:", font=large_font).grid(column=0, row=2, sticky="w")
    tk.Label(infoFr, text="Rate:", font=large_font).grid(column=0, row=3, sticky="w")

    # Entry boxes for the info frame
    tk.Entry(infoFr, textvariable=subject_var, font=large_font, width=25).grid(column=1, row=0, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=trial_var, font=large_font, width=25).grid(column=1, row=1, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=task_var, font=large_font, width=25).grid(column=1, row=2, padx=5, pady=5, sticky="ew")
    tk.Entry(infoFr, textvariable=rate_var, font=large_font, width=25).grid(column=1, row=3, padx=5, pady=5, sticky="ew")

    # Submit Button
    tk.Button(infoFr, text='Submit', command=submitData, font=large_font, width=10).grid(column=0, row=4, columnspan=2, pady=10)

    # Pedal Indicators Frame
    pedalFr = tk.LabelFrame(win, text="Pedal Indicators", padx=15, pady=15, font=large_font_bold)
    pedalFr.grid(column=0, row=1, padx=10, pady=10, sticky="nsew")

    pedals = [
        ("Upper Left", 0, 2, 1), ("Upper Right", 0, 3, 1),
        ("Lower Left", 1, 2, 1), ("Lower Right", 1, 3, 1),
        ("Clutch", 0, 1, 1), ("Camera", 1, 1, 1), ("Long", 0, 0, 3)
    ]

    for pedal in pedals:
        text, row, col, rowspan = pedal
        tk.Label(pedalFr, text=text, bg="red", width=9, height=4, font=large_font).grid(row=row, column=col, padx=10, pady=10, rowspan=rowspan)

    # Diagnostic Frame
    diagnosticFr = tk.LabelFrame(win, text="Diagnostics", padx=15, pady=15, font=large_font_bold)
    diagnosticFr.grid(column=1, row=0, rowspan=2, padx=10, pady=10, sticky="nsew")

    devices = [
        ("Capture 1:", "Error"), ("Capture 2:", "Error"),
        ("Watch Left:", "Error"), ("Watch Right:", "Error"),
        ("Arduino (Pedals):", "Error")
    ]

    for i, (label, status) in enumerate(devices):
        tk.Label(diagnosticFr, text=label, font=large_font).grid(row=i, column=0, padx=10, pady=5, sticky="w")
        tk.Label(diagnosticFr, text=status, bg="red", width=15, font=large_font).grid(row=i, column=1, padx=10, pady=5, sticky="w")

    # Configure weight for responsiveness
    for i in range(2):
        win.grid_columnconfigure(i, weight=1)
    for i in range(3):
        win.grid_rowconfigure(i, weight=1)

    win.mainloop()

def submitData():
    from app import sendData
    sendData(subject_var.get(), trial_var.get(), task_var.get(), rate_var.get())