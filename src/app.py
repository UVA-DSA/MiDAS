import eel
import subprocess
import datetime
from os import mkdir
import csv
from Video.capture import send_vid_data 
from queue import Queue
from threading import Thread

#defines
capture_q_size = 10

#global variables
capture_q = Queue()

eel.init('web')

@eel.expose
def sendData(subject, trial, task, rate, range_val, hemisphere, filter_val):
    '''
    Takes data in from frontend, puts it in a dictionary, and passes dictionary into function 
    getInputStreams(). All data values come in as strings and are converted into respective 
    types.

    subject: Name of subject running trial
    trial: Trial number as an integer
    task: Name of task being performed
    rate: Rate in Hz as a float
    range_val: Range between 36, 72, and 144 in
    hemisphere: front, back, left, right, top, or bottom
    filter_val: wide or narrow.

    returns: Nothing
    '''
    data = {
        "Subject" : subject,
        "Trial": int(trial),
        "Task": task,
        "Rate": float(rate),
        "Range": float(range_val),
        "Hemisphere": hemisphere,
        "Filter": filter_val,
        "Date": str(datetime.datetime.today().date()),
        "Time": str(datetime.datetime.today().time())[:8]
        }
    print(data)
    path = f"./../{data['Task']}_S{data['Subject']}_T{data['Trial']}_{data['Date']}_{data['Time']}" 
    mkdir(path)
    getInputStreams(data)
    return

def readData(capture_q): #add readKinematic, etc flags in future
    if (capture_q != None):
        capture_data = capture_q.get()

def startThreads(data: dict):
    '''
    Reaches out to different data streams (socket) to get data. Prepares to write them to file 

    data: dict that contains user data

    returns: Nothing
    '''
    capture_thread = Thread(target = send_vid_data, args=(capture_q, capture_q_size,))
    read_thread = Thread(target = readData, args = (capture_q, ))
    #future threads go here 

    #start threads
    capture_thread.start()
    read_thread.start()

    #wait for join
    capture_thread.join()
    read_thread.join()

    return

eel.start('index.html')