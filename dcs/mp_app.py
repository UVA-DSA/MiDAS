import datetime
import os 
import subprocess
import csv
import newGUI
import threading
from Video.capture import send_vid_data 
from Video.camera import get_camera_handler
from Trakstar.trackstarWriter import get_trakstar_data, exec_trakstar
# from Smartwatch.tcp_smartwatch_client import receive_smartwatch_data
from Smartwatch.bluetooth_smartwatch_receiver import receive_smartwatch_data
from PDS.PDS import get_PDS_data
from multiprocessing import Process, Queue, Event
from multiprocessing.managers import BaseManager
from time import sleep
from queue import LifoQueue
from config import smartwatch_1_id, smartwatch_1_ip,smartwatch_2_id,smartwatch_2_ip,smartwatch_port,camera_type_1, camera_type_2
from Audio.audio_capture import capture_audio_transmit
# from RAVENPackets.get_RAVEN_packets import start_logger

import time
import sys



# defines
CAPTURE_Q_SIZE = 100
AUDIO_Q_SIZE = 1024
TRACKSTAR_Q_SIZE = 128
SMARTWATCH_Q_SIZE = 2048
PDS_Q_SIZE = 100
GUI_Q_SIZE = 100
RAVEN_Q_SIZE = 100

def run(lifo):
    # get next message or wait until one is available
    s = lifo.get(block=True)
    print(s)


# create manager that knows how to create and manage LifoQueues
class MyManager(BaseManager):
    pass
MyManager.register('LifoQueue', LifoQueue)

thread_stop = Event()

def killAllProcesses():
    global thread_stop
    thread_stop.set()
    


def sendData(subject, trial, task, rate, watch_toggle):
    thread_stop.clear()
    
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

    print(path)

    send_data_process = threading.Thread(target=startProcesses, args=(path, rate, task, watch_toggle))
    send_data_process.start()
    return 

def endProgram():
    global thread_stop
    print("Program ended")
    thread_stop.set()

def readData(task, rate, list_of_qs, path, thread_stop):
    if rate == 0:
        print("Rate not valid")
        exit(-1)

    sample_time = 1 / int(rate)
    print("Sample rate(hz), time(s): ", rate, sample_time)

    csv_path = path + '/synced_data.csv'
    

    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["server_time",
                            "trakstar_SensorID", "trakstar_Status","trakstar_x", "trakstar_y", "trakstar_z", "trakstar_azimuth", "trakstar_elevation", "trakstar_roll","trakstar_time", 
                            "trakstar_SensorID", "trakstar_Status","trakstar_x", "trakstar_y", "trakstar_z", "trakstar_azimuth", "trakstar_elevation", "trakstar_roll","trakstar_time",
                            "trakstar_SensorID", "trakstar_Status","trakstar_x", "trakstar_y", "trakstar_z", "trakstar_azimuth", "trakstar_elevation", "trakstar_roll","trakstar_time",
                            "trakstar_SensorID", "trakstar_Status","trakstar_x", "trakstar_y", "trakstar_z", "trakstar_azimuth", "trakstar_elevation", "trakstar_roll", "trakstar_time",
                            "video_time", "video_id",
                            "intel_time", "intel_frame",
                            'smartwatch_1_time','smartwatch_1_wrist_position','smartwatch_1_sensor_type','smartwatch_1_value_X_Axis','smartwatch_1_value_Y_Axis','smartwatch_1_value_Z_Axis',
                            'smartwatch_2_time','smartwatch_2_wrist_position','smartwatch_2_sensor_type','smartwatch_2_value_X_Axis','smartwatch_2_value_Y_Axis','smartwatch_2_value_Z_Axis',
                            'PDS_time','pedal_1', 'pedal_2', 'pedal_3', 'pedal_4', 'pedal_5', 'pedal_6', 'pedal_7', 'pedal_1_pressed', 'pedal_2_pressed', 'pedal_3_pressed', 'pedal_4_pressed', 'pedal_5_pressed', 'pedal_6_pressed', 'pedal_7_pressed',
                            'zed_time', 'zed_frame'])
    
        while True:
            if thread_stop.is_set():
                print("[Main Data Saver: Thread stop set, exiting..]")
                csv_file.close()
                break
            try:
                start_time = time.time()
                local_time = time.ctime(start_time)
                #get current epoch time in ns
                local_time = int(time.time_ns())
                
                trackstar_data = None
                camera_data_intel = None
                camera_data_zed = None
                smartwatch_1_data = None
                smartwatch_2_data = None
                video_data = None
                PDS_data = None
        
                try:
                    video_data = list_of_qs[0].get(block=False)
                except:
                    pass

                try:
                    camera_data_intel = list_of_qs[1].get(block=False)
                except:
                    pass

                try:
                    camera_data_zed = list_of_qs[-1].get(block=False)
                except:
                    pass
                    
                try:
                    trackstar_data = list_of_qs[2].get(block=False)
                    # print("[Read Process: TrakStar Data: ]",trackstar_data)
                except:
                    pass
                
                
                try:
                    smartwatch_1_data = list_of_qs[3].get(block=False)
                    # print("[Read Process: Smartwatch 1 Data: ]",smartwatch_1_data)
                except:
                    pass
                
                
                try:
                    smartwatch_2_data = list_of_qs[4].get(block=False)
                except:
                    pass

                try:
                    PDS_data = list_of_qs[5].get(block=False)
                except:
                    pass
                
                # Add logic to handle the case where trackstar_data or video_data is None
                # if video_data is None or smartwatch_data is None or trackstar_data is None:
                #     continue
                
                if video_data is None:
                    video_data = [0, 0, 0]

                if camera_data_intel is None:
                    camera_data_intel = [-1, -1]

                if camera_data_zed is None:
                    camera_data_zed = [-1, -1]
                    
                if smartwatch_1_data is None:
                    smartwatch_1_data = [0,0,0,0,0,0]
                    
                if smartwatch_2_data is None:
                    smartwatch_2_data = [0,0,0,0,0,0]
                    
                if trackstar_data is None:
                    trackstar_data = [0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0]
                
                if PDS_data is None:
                    PDS_data = [-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]
                
                # print("Writing to csv file ..")
                collectedData = [local_time, trackstar_data[0], trackstar_data[1], trackstar_data[2], trackstar_data[3], trackstar_data[4], trackstar_data[5], trackstar_data[6], trackstar_data[7], trackstar_data[8], trackstar_data[9], trackstar_data[10], trackstar_data[11], trackstar_data[12], trackstar_data[13], trackstar_data[14], trackstar_data[15], trackstar_data[16], trackstar_data[17], trackstar_data[18], trackstar_data[19], trackstar_data[20], trackstar_data[21], trackstar_data[22], trackstar_data[23], trackstar_data[24], trackstar_data[25], trackstar_data[26], trackstar_data[27], trackstar_data[28], trackstar_data[29], trackstar_data[30], trackstar_data[31], trackstar_data[32], trackstar_data[33], trackstar_data[34], trackstar_data[35],video_data[0], video_data[1], camera_data_intel[0], camera_data_intel[1], smartwatch_1_data[0],smartwatch_1_data[1],smartwatch_1_data[2],smartwatch_1_data[3],smartwatch_1_data[4],smartwatch_1_data[5] , smartwatch_2_data[0],smartwatch_2_data[1],smartwatch_2_data[2],smartwatch_2_data[3],smartwatch_2_data[4],smartwatch_2_data[5],PDS_data[0],PDS_data[1],PDS_data[2],PDS_data[3],PDS_data[4],PDS_data[5],PDS_data[6],PDS_data[7],PDS_data[8],PDS_data[9],PDS_data[10],PDS_data[11],PDS_data[12],PDS_data[13],PDS_data[14], camera_data_zed[0], camera_data_zed[1]]
                csv_writer.writerow(collectedData)
                csv_file.flush()
                
                
                try:
                    list_of_qs[6].put(collectedData, block=False)
                except:
                    while not list_of_qs[6].empty():
                        list_of_qs[6].get()
                
                
                # sleep_time = sample_time - (time.time() - start_time)
                while (time.time() - start_time) < sample_time:
                    pass


            except KeyboardInterrupt:
                print("Keyboard Interrupt!")
                exit(-1)



def startProcesses(path, rate, task, watch_toggle):
    
    manager = MyManager()
    manager.start()
    
    camera_q_intel = manager.LifoQueue(CAPTURE_Q_SIZE)
    camera_q_zed = manager.LifoQueue(CAPTURE_Q_SIZE)
    capture_q = manager.LifoQueue(CAPTURE_Q_SIZE)
    depth_img_q = manager.LifoQueue(CAPTURE_Q_SIZE)
    zed_img_q = manager.LifoQueue(CAPTURE_Q_SIZE)
    OBS_img_q = manager.LifoQueue(CAPTURE_Q_SIZE)
    trackstar_q = manager.LifoQueue(TRACKSTAR_Q_SIZE)
    smartwatch_1_q = manager.LifoQueue(SMARTWATCH_Q_SIZE)
    smartwatch_2_q = manager.LifoQueue(SMARTWATCH_Q_SIZE)
    PDS_q = manager.LifoQueue(PDS_Q_SIZE)
    gui_q = manager.LifoQueue(GUI_Q_SIZE)
    audio_q = manager.LifoQueue(AUDIO_Q_SIZE)
    RAVEN_q = manager.LifoQueue(RAVEN_Q_SIZE)

    list_of_qs = [capture_q, camera_q_intel, trackstar_q, smartwatch_1_q, smartwatch_2_q, PDS_q, gui_q, depth_img_q, OBS_img_q, audio_q, zed_img_q, camera_q_zed]

    video_capture_process = Process(name="OBS Virtual Camera Capture", target=send_vid_data, args=(capture_q, path, OBS_img_q, thread_stop))
    video_capture_process.daemon = True

    camera_handler_intel = get_camera_handler(camera_type_1)
    camera_capture_process = Process(name="Depth Camera Intel Capture", target=camera_handler_intel, args=(camera_q_intel, path, depth_img_q, thread_stop))
    camera_capture_process.daemon = True

    camera_handler_zed = get_camera_handler(camera_type_2)
    camera_capture_process_zed = Process(name="Depth Camera Zed Capture", target=camera_handler_zed, args=(camera_q_zed, path, zed_img_q, thread_stop))
    camera_capture_process_zed.daemon = True
    
    trakstar_process = Process(name="TrakStar Capture", target=get_trakstar_data, args=(trackstar_q, path, thread_stop))
    trakstar_process.daemon = True


    trakstar_exec_process = Process(name="TrakStar Execution", target=exec_trakstar, args=(thread_stop,))
    trakstar_exec_process.daemon = True

    smartwatch_1_process = Process(name="Smartwatch Left Capture", target=receive_smartwatch_data, args=(smartwatch_1_ip,smartwatch_port,smartwatch_1_q,path, smartwatch_1_id, thread_stop))
    smartwatch_1_process.daemon = True
    
    smartwatch_2_process = Process(name="Smartwatch Right Capture", target=receive_smartwatch_data, args=(smartwatch_2_ip,smartwatch_port,smartwatch_2_q,path, smartwatch_2_id, thread_stop))
    smartwatch_2_process.daemon = True
    
    read_process = Process(name="Main Data Saver", target=readData, args=(task, rate, list_of_qs, path, thread_stop))
    read_process.daemon = True

    PDS_process = Process(name="PDS Capture", target=get_PDS_data, args = (PDS_q, path, thread_stop))
    PDS_process.daemon = True

    
    # audio_recorder_process = Process(name="Audio Capture", target=capture_audio_transmit, args=(audio_q, path, thread_stop))
    # audio_recorder_process.daemon = True

    # RAVEN_process = Process(name="RAVEN Capture", target=start_logger, args = (RAVEN_q, path, thread_stop))
    # RAVEN_process.daemon = True

    #Has to be thread since shares memory with GUI, has to be on same process
    updateIndicator_process = threading.Thread(target=newGUI.updateIndicators, args=(gui_q,depth_img_q, OBS_img_q,zed_img_q, thread_stop))
    updateIndicator_process.daemon = True

    # camera_capture_process - add this to enable depth cam
    #  smartwatch_1_process, smartwatch_2_process, camera_capture_process
    # processes = [camera_capture_process_zed, video_capture_process, trakstar_process,read_process, PDS_process, trakstar_exec_process]
    processes = [video_capture_process, read_process]
    threads = [updateIndicator_process]

    if(watch_toggle):
        print("[Main Program] Enabling smartwatch processes")
        processes.append(smartwatch_1_process)
        processes.append(smartwatch_2_process)


    for process in processes:
        process.start()

    for thread in threads:
        thread.start()

    print("[Main Program: Processes started]")
    try:
        for process in processes:
            process.join()
            
        print("[Main Program: All processes have completed]")
        
        print("[Main Program: Closing all queues]")
        
        for q in list_of_qs:
            q = None
            
            
    except KeyboardInterrupt:
        # camera_handler.finish()
        print("KeyboardInterrupt received. Exiting main program.")
        for process in processes:
            process.terminate()
        exit(-1)

    finally:
        manager.shutdown()
        
    print("[Main Program: All Resources Released!.]")
    print("[Main Program: Ready for Recording!.]")
    return

if __name__ == "__main__":
    # eel.start('index.html')
    newGUI.startGui()