import eel
import datetime
import os 
import csv
from Video.capture import send_vid_data 
from Trakstar.trackstarWriter import get_trakstar_data
from PDS.PDS import get_PDS_data
import newGUI

from queue import Queue
from threading import Thread, Event
import threading
import time
import sys


#defines
CAPTURE_Q_SIZE = 10
TRACKSTAR_Q_SIZE = 100
PDS_Q_SIZE = 100


#global variables
capture_q = Queue(CAPTURE_Q_SIZE)
trackstar_q = Queue(TRACKSTAR_Q_SIZE)
PDS_q = Queue(PDS_Q_SIZE)
thread_stop = True

#eel.init('web')

#@eel.expose
def sendData(subject, trial, task, rate):
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
        "Task": str(task),
        "Rate": float(rate),
        "Date": str(datetime.datetime.today().date()),
        "Time": str(datetime.datetime.today().time())[:8]
        }
    print(data)
    path = f"{data['Task']}_S{data['Subject']}_T{data['Trial']}_{data['Date']}/" 
    
    mydir = "./src/Data/"
    myfile = path
    path = os.path.join(mydir, myfile)

    
    if not os.path.exists(path):
        os.makedirs(path)
    
    print(path)
    send_data_thread = threading.Thread(target=startThreads, args=(path, rate, task))
    send_data_thread.start()
    return 

# @eel.expose
def endProgram():
    global thread_stop
    print("Program ended")
    thread_stop = False

def readData(task, rate, list_of_qs): #add readKinematic, etc flags in future
    global thread_stop
    if rate == 0:
        print("Rate not valid")
        exit(-1)

    sample_time = 1/int(rate)
    print("Sample rate(hz),time(s): ",rate,sample_time)

    # try:
    #     while True:
    #         start_time = time.time()
    #         # Your existing code here

    #         # Make sure to sleep for the sample time
    #         sleep_time = sample_time - (time.time() - start_time)
    #         if sleep_time > 0:
    #             time.sleep(sleep_time)

    # except KeyboardInterrupt:
    #     print("KeyboardInterrupt received. Exiting threads.")
    #     # Perform any cleanup here if necessary

    csv_path = './Data/' + task + 'TrackStar+VideoData.csv' 
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Trackstar Data + Video Data'])
    #time.sleep(float(10)) #adding this to give time for the video capture card to get up and runnning and then start processing data
    with open(csv_path, 'w', newline = "") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Trackstar Data + Video Data'])
        while thread_stop:
            try:
                start_time = time.time()
                local_time = time.ctime(start_time)
                print("Adding data to csv")
                #trackstar_q = list_of_qs[1].get().split(',')
                #csv_writer.writerow([local_time,start_time, trackstar_q[0], trackstar_q[1],trackstar_q[2], trackstar_q[3], trackstar_q[4], trackstar_q[5], trackstar_q[6], trackstar_q[7], list_of_qs[0].get()[3]])
                trackstar_data = list_of_qs[1].get()
                csv_writer.writerow([local_time,start_time, trackstar_data[0], trackstar_data[1], trackstar_data[2], trackstar_data[3], trackstar_data[4], trackstar_data[5], trackstar_data[6], trackstar_data[7], trackstar_data[8], trackstar_data[3]])
                #csv_writer.writerow([local_time,start_time, list_of_qs[1].get()[0], list_of_qs[1].get()[1], list_of_qs[1].get()[2], list_of_qs[1].get()[3], list_of_qs[1].get()[4], list_of_qs[1].get()[5], list_of_qs[1].get()[6], list_of_qs[1].get()[7], list_of_qs[1].get()[8], list_of_qs[0].get()[3]])
                csv_file.flush()
                # for idx,q in enumerate(list_of_qs):
                #     if(idx == 0):
                #         print("Video Capture Queue:")
                #     if(idx == 1):
                #         print("Trackstar Queue:")
                        
                #     # if(not q.empty()):
                #     #     print(q.queue[-1])

                #make sure sleep for sample time
                sleep_time = sample_time - (time.time() - start_time)
                if(sleep_time > 0):
                    time.sleep(sleep_time)
            except KeyboardInterrupt:
                print("Keyboard Interrupt!")
                exit(-1)
        else:
           return

def startThreads(path, rate, task):

    list_of_qs = [capture_q, trackstar_q, PDS_q]
    print(path)
    capture_thread = Thread(target = send_vid_data, args=(capture_q, path))
    capture_thread.daemon = True
    trakstar_thread = Thread(target = get_trakstar_data, args=(trackstar_q, path ))
    trakstar_thread.daemon = True
    read_thread = Thread(target = readData, args = (task, rate, list_of_qs))
    read_thread.daemon = True
    PDS_thread = Thread(target=get_PDS_data, args = (PDS_q, path))
    PDS_thread.daemon = True
    updateIndicator_thread = Thread(target=newGUI.updateIndicators, args = (PDS_q, capture_q,))
    updateIndicator_thread.daemon = True
    
    #future threads go here 

    #start threads
    capture_thread.start()
    trakstar_thread.start()
    # read_thread.start()
    PDS_thread.start()
    updateIndicator_thread.start()

    try:
        # Wait for threads to finish
        capture_thread.join()
        trakstar_thread.join()
        # read_thread.join()
        PDS_thread.join()
        updateIndicator_thread.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting main program.")
        exit(-1)

    return

#eel.start('index.html')
newGUI.startGui()