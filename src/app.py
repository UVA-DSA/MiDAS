import eel
import socket
import subprocess
import pickle
import datetime
from os import mkdir
import csv
import cv2

#defines
video_port = 12345
video_stream_size = 4096
trackstar_port = 12346
trackstar_stream_size = 4096
raven_port = 12347
raven_stream_size = 4096

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

def getInputStreams(data: dict):
    '''
    Reaches out to different data streams (socket) to get data. Prepares to write them to file 

    data: dict that contains user data

    returns: Nothing
    '''
    
    #Starting subprocesses (other files)
    subprocess.Popen(['python3', 'video/capture.py']) 
    path = f"./../../{data['Task']}_S{data['Subject']}_T{data['Trial']}_{data['Date']}_{data['Time']}"
    subprocess.Popen(['./Trackstar/trackStarCollector', data['Rate'], data['Range'], data['Filter'], data['Hemisphere'], path])
    
    path = f"./../{data['Task']}_S{data['Subject']}_T{data['Trial']}_{data['Date']}_{data['Time']}" 

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as video, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as trackstar, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as raven:
        video.bind(('localhost', video_port))
        video.listen()

        trackstar.bind(('localhost', trackstar_port))
        trackstar.listen()

        raven.bind(('NEED TO FILL OUT', raven_port)) #TODO: Fill out proper IP
        raven.listen()

        #setting up video connection
        vid_con, _ = video.accept()
        fourcc = cv2.VideoWriter_fourcc(* "mp4v")
        out = cv2.VideoWriter(str(path) + "/video"  + ".mp4", fourcc, 30.0, (640, 480))
        with open(f"{path}/frames.csv", "a") as video_frames_file: #writing data to file
                writer = csv.writer(video_frames_file)
                writer.writerow("Frame #", "Time Stamp")
        print("Connected to Video")

        #setting up trackstar connection
        trackstar_con, _ = trackstar.accept()
        print("Connected to TrackStar")

        #setting up raven connection
        raven_con, _ = raven.accept()
        with open(f"{path}/raven.csv", "a") as video_frames_file: #writing data to file
                writer = csv.writer(video_frames_file)
                raven_header = []#TODO: FIll out with neccessary
                writer.writerow(raven_header)
        print("Connected to RAVEN")

        

        while True:

            #video data processing
            video_data = pickle.loads(vid_con.recv(video_stream_size))
            if not video_data:
                pass
            if video_data == "Data not collected": #something wrong with video data
                pass #TODO: update later with what to do
            out.write(video_data[0])
            with open(f"{path}/frames.csv", "a") as video_frames_file: #writing data to file
                writer = csv.writer(video_frames_file)
                writer.writerow(video_data[1], video_data[2])



            #trackstar data processing
            trackstar_data = trackstar_con.recv(trackstar_stream_size).decode()
            if not trackstar_data:
                pass
            sensorID, status, x, y, z, azimuth, elevation, roll, time, quality = trackstar_data.split(',')
            with open(f"{path}/trackstar.csv", "a") as trackstar_file: #writing data to file
                writer = csv.writer(trackstar_file)
                writer.writerow([sensorID, status, x, y, z, azimuth, elevation, roll, time, quality])

            
            #trackstar data processing
            raven_data = pickle.loads(raven_con.recv(raven_stream_size))
            if not raven_data:
                pass
            raven_data_as_list = raven_data.split(',') #TODO: Fill out neccessary headers
            with open(f"{path}/raven.csv", "a") as raven_file: #writing data to file
                writer = csv.writer(raven_file)
                writer.writerow(raven_data_as_list)
            
              
    return

eel.start('index.html')