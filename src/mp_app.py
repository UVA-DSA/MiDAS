import eel
import datetime
import os 
import csv
from Video.capture import send_vid_data 
from Trakstar.trackstarWriter import get_trakstar_data
from Smartwatch.tcp_smartwatch_client import receive_smartwatch_data
from multiprocessing import Process, Queue, Event
import time
import sys

# defines
CAPTURE_Q_SIZE = 10
TRACKSTAR_Q_SIZE = 100
SMARTWATCH_Q_SIZE = 100

# global variables
capture_q = Queue(CAPTURE_Q_SIZE)
trackstar_q = Queue(TRACKSTAR_Q_SIZE)
smartwatch_q = Queue(SMARTWATCH_Q_SIZE)

thread_stop = Event()

# Replace these values with your smartwatch's IP, port
smartwatch_ip = '127.0.0.1'
smartwatch_port = 7889

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

    returns: Nothing
    '''
    data = {
        "Subject": subject,
        "Trial": int(trial),
        "Task": str(task),
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

    send_data_process = Process(target=startProcesses, args=(path, rate, task))
    send_data_process.start()
    return 

@eel.expose
def endProgram():
    global thread_stop
    print("Program ended")
    thread_stop.set()

def readData(task, rate, list_of_qs):
    global thread_stop
    if rate == 0:
        print("Rate not valid")
        exit(-1)

    sample_time = 1 / int(rate)
    print("Sample rate(hz), time(s): ", rate, sample_time)

    csv_path = './Data/' + task + 'TrackStar+VideoData.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Trackstar Data + Video Data'])
    
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Trackstar Data + Video Data'])
        while not thread_stop.is_set():
            try:
                start_time = time.time()
                local_time = time.ctime(start_time)
                
                trackstar_data = None
                smartwatch_data = None
                video_data = None
        
                try:
                    video_data = list_of_qs[0].get(block=True)
                except:
                    pass
  
                    
                # try:
                #     trackstar_data = list_of_qs[1].get(block=False)
                # except:
                #     pass
                
                
                try:
                    smartwatch_data = list_of_qs[2].get(block=True)
                except:
                    pass
                
                # Add logic to handle the case where trackstar_data or video_data is None
                if video_data is None  or smartwatch_data is None:
                    continue
                
                trackstar_data = [''] * 9  # Adjust the number of elements as needed

                
                # print("Writing to csv file ..")
                csv_writer.writerow([local_time, start_time, trackstar_data[0], trackstar_data[1], trackstar_data[2], trackstar_data[3], trackstar_data[4], trackstar_data[5], trackstar_data[6], trackstar_data[7], trackstar_data[8], video_data[0], video_data[1], video_data[-1], smartwatch_data[0],smartwatch_data[1]])
                csv_file.flush()
                
                
                # sleep_time = sample_time - (time.time() - start_time)
                while (time.time() - start_time) < sample_time:
                    pass 

            except KeyboardInterrupt:
                print("Keyboard Interrupt!")
                exit(-1)
        else:
            return

def startProcesses(path, rate, task):
    list_of_qs = [capture_q, trackstar_q, smartwatch_q]

    capture_process = Process(target=send_vid_data, args=(capture_q, path))
    capture_process.daemon = True
    
    trakstar_process = Process(target=get_trakstar_data, args=(trackstar_q, path))
    trakstar_process.daemon = True
    
    smartwatch_process = Process(target=receive_smartwatch_data, args=(smartwatch_ip,smartwatch_port,smartwatch_q,path))
    smartwatch_process.daemon = True
    
    read_process = Process(target=readData, args=(task, rate, list_of_qs))
    read_process.daemon = True

    capture_process.start()
    trakstar_process.start()
    smartwatch_process.start()

    read_process.start()

    try:
        capture_process.join()
        trakstar_process.join()
        smartwatch_process.join()
        read_process.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting main program.")
        exit(-1)

    return

if __name__ == "__main__":
    eel.start('index.html')