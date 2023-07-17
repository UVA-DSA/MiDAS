import eel
import datetime
import os 
from Video.capture import send_vid_data 
from Trakstar.trackstarWriter import getTrackStarData


from queue import Queue
from threading import Thread
import time


#defines
CAPTURE_Q_SIZE = 10
TRACKSTAR_Q_SIZE = 100


#global variables
capture_q = Queue(CAPTURE_Q_SIZE)
trackstar_q = Queue(TRACKSTAR_Q_SIZE)

eel.init('web')

@eel.expose
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
        "Task": task,
        "Rate": float(rate),
        "Date": str(datetime.datetime.today().date()),
        "Time": str(datetime.datetime.today().time())[:8]
        }
    print(data)
    path = f"{data['Task']}_S{data['Subject']}_T{data['Trial']}_{data['Date']}/" 
    
    mydir = "./Data/"
    myfile = path
    path = os.path.join(mydir, myfile)

    
    if not os.path.exists(path):
        os.makedirs(path)
    
    
    startThreads(path, rate)
    return

def readData(rate, list_of_qs): #add readKinematic, etc flags in future

    sample_time = 1/int(rate)
    print("Sample rate(hz),time(s): ",rate,sample_time)

    while True:
        
        try:
            
            start_time = time.time()
            for idx,q in enumerate(list_of_qs):
                if(idx == 0):
                    print("Video Capture Queue:")
                if(idx == 1):
                    print("Trackstar Queue:")
                    
                if(not q.empty()):
                    print(q.queue[-1])

            #make sure sleep for sample time
            sleep_time = sample_time - (time.time() - start_time)
            if(sleep_time > 0):
                time.sleep(sleep_time)
    
        except Exception as e:
            print("Exception!", e)

def startThreads(path, rate):

    list_of_qs = [capture_q, trackstar_q]

    capture_thread = Thread(target = send_vid_data, args=(capture_q, path))
    capture_thread.daemon = True
    trakstar_thread = Thread(target = getTrackStarData, args=(trackstar_q, path ))
    trakstar_thread.daemon = True
    read_thread = Thread(target = readData, args = (rate, list_of_qs))
    read_thread.daemon = True
    
    #future threads go here 

    #start threads
    capture_thread.start()
    trakstar_thread.start()
    read_thread.start()

    #wait for join
    capture_thread.join()
    trakstar_thread.join()
    read_thread.join()

    return

eel.start('index.html')