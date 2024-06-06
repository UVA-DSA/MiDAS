import eel
import datetime
import os 
import csv
from Video.capture import send_vid_data 
from Video.camera import get_camera_instance
from Trakstar.trackstarWriter import get_trakstar_data
from Smartwatch.tcp_smartwatch_client import receive_smartwatch_data
from multiprocessing import Process, Queue, Event
from multiprocessing.managers import BaseManager
from time import sleep
from queue import LifoQueue

import time
import sys

# defines
CAPTURE_Q_SIZE = 10
TRACKSTAR_Q_SIZE = 100
SMARTWATCH_Q_SIZE = 100

# global variables
# capture_q = LifoQueue(CAPTURE_Q_SIZE)
# trackstar_q = LifoQueue(TRACKSTAR_Q_SIZE)
# smartwatch_1_q = LifoQueue(SMARTWATCH_Q_SIZE)
# smartwatch_2_q = LifoQueue(SMARTWATCH_Q_SIZE)

def run(lifo):
    # get next message or wait until one is available
    s = lifo.get(block=True)
    print(s)


# create manager that knows how to create and manage LifoQueues
class MyManager(BaseManager):
    pass
MyManager.register('LifoQueue', LifoQueue)


camera_type = "Intel"

thread_stop = Event()

# Replace these values with your smartwatch's IP, port
smartwatch_1_ip = '172.27.191.158'
smartwatch_1_id = 'right'

smartwatch_2_ip = '172.27.176.73'
smartwatch_2_id = 'left'

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

def readData(task, rate, list_of_qs, path):
    global thread_stop
    if rate == 0:
        print("Rate not valid")
        exit(-1)

    sample_time = 1 / int(rate)
    print("Sample rate(hz), time(s): ", rate, sample_time)

    csv_path = path + '/synced_data.csv'
    

    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["server_time",  "trakstar_SensorID", "trakstar_Status","trakstar_x", "trakstar_y", "trakstar_z", "trakstar_azimuth", "trakstar_elevation", "trakstar_roll", "trakstar_atime", "video_time", "video_id", "3d_camera_time", "3d_camera_frame_id", 'smartwatch_1_time','smartwatch_1_wrist_position','smartwatch_1_sensor_type','smartwatch_1_value_X_Axis','smartwatch_1_value_Y_Axis','smartwatch_1_value_Z_Axis', 'smartwatch_2_time','smartwatch_2_wrist_position','smartwatch_2_sensor_type','smartwatch_2_value_X_Axis','smartwatch_2_value_Y_Axis','smartwatch_2_value_Z_Axis'])
    
        while not thread_stop.is_set():
            try:
                start_time = time.time()
                local_time = time.ctime(start_time)
                #get current epoch time in ns
                local_time = int(time.time_ns())
                
                trackstar_data = None
                camera_data = None
                smartwatch_1_data = None
                smartwatch_2_data = None
                video_data = None
        
                try:
                    video_data = list_of_qs[0].get(block=False)
                except:
                    pass

                try:
                    camera_data = list_of_qs[1].get(block=False)
                except:
                    pass
  
                    
                try:
                    trackstar_data = list_of_qs[2].get(block=False)
                except:
                    pass
                
                
                try:
                    smartwatch_1_data = list_of_qs[3].get(block=False)
                except:
                    pass
                
                
                try:
                    smartwatch_2_data = list_of_qs[4].get(block=False)
                except:
                    pass
                
                
                # Add logic to handle the case where trackstar_data or video_data is None
                # if video_data is None or smartwatch_data is None or trackstar_data is None:
                #     continue
                
                if video_data is None:
                    video_data = [0, 0, 0]

                if camera_data is None:
                    camera_data = [0, 0]
                    
                if smartwatch_1_data is None:
                    smartwatch_1_data = [0,0,0,0,0,0]
                    
                if smartwatch_2_data is None:
                    smartwatch_2_data = [0,0,0,0,0,0]
                    
                if trackstar_data is None:
                    trackstar_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                
                
                # print("Writing to csv file ..")
                csv_writer.writerow([local_time, trackstar_data[0], trackstar_data[1], trackstar_data[2], trackstar_data[3], trackstar_data[4], trackstar_data[5], trackstar_data[6], trackstar_data[7], trackstar_data[8], video_data[0], video_data[1], camera_data[0], camera_data[1], smartwatch_1_data[0],smartwatch_1_data[1],smartwatch_1_data[2],smartwatch_1_data[3],smartwatch_1_data[4],smartwatch_1_data[5] , smartwatch_2_data[0],smartwatch_2_data[1],smartwatch_2_data[2],smartwatch_2_data[3],smartwatch_2_data[4],smartwatch_2_data[5]])
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
    
    manager = MyManager()
    manager.start()
    
    camera_q = manager.LifoQueue(CAPTURE_Q_SIZE)
    capture_q = manager.LifoQueue(CAPTURE_Q_SIZE)
    trackstar_q = manager.LifoQueue(TRACKSTAR_Q_SIZE)
    smartwatch_1_q = manager.LifoQueue(SMARTWATCH_Q_SIZE)
    smartwatch_2_q = manager.LifoQueue(SMARTWATCH_Q_SIZE)

    list_of_qs = [capture_q, trackstar_q, smartwatch_1_q, smartwatch_2_q]

    video_capture_process = Process(target=send_vid_data, args=(capture_q, path))
    video_capture_process.daemon = True

    camera_handler = get_camera_instance(camera_type)(camera_q, path)
    camera_capture_process = Process(target=camera_handler.run, args=())
    camera_capture_process.daemon = True
    
    trakstar_process = Process(target=get_trakstar_data, args=(trackstar_q, path))
    trakstar_process.daemon = True
    
    smartwatch_1_process = Process(target=receive_smartwatch_data, args=(smartwatch_1_ip,smartwatch_port,smartwatch_1_q,path, smartwatch_1_id))
    smartwatch_1_process.daemon = True
    
    smartwatch_2_process = Process(target=receive_smartwatch_data, args=(smartwatch_2_ip,smartwatch_port,smartwatch_2_q,path, smartwatch_2_id))
    smartwatch_2_process.daemon = True
    
    read_process = Process(target=readData, args=(task, rate, list_of_qs, path))
    read_process.daemon = True

    video_capture_process.start()
    camera_capture_process.start()
    trakstar_process.start()
    
    smartwatch_1_process.start()
    smartwatch_2_process.start()

    read_process.start()

    try:
        video_capture_process.join()
        camera_capture_process.join()
        trakstar_process.join()
        smartwatch_1_process.join()
        smartwatch_2_process.join()
        
        read_process.join()

    except KeyboardInterrupt:
        camera_handler.finish()
        print("KeyboardInterrupt received. Exiting main program.")
        exit(-1)

    return

if __name__ == "__main__":
    eel.start('index.html')