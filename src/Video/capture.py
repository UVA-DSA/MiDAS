from distutils.log import error
from posixpath import dirname
from unicodedata import name
import cv2  #for capturing
from matplotlib import pyplot as plt #for displaying
import os #to create directory
import json #to read JSON file and create the directory
import time
from colorama import Fore
import sys
import shutil
import pandas as pd
import numpy as np
import csv
import ffmpeg


def error_handler(text, directory):
    print(Fore.YELLOW + text)
    
    try:
        cap.realease()
    except: 
        pass
    
    try:
        cap.destroyAllWinidows()
    except: 
        pass
    
    try:
        out.release()
    except:
        pass
    try:
        os.remove("updateFile.txt")
    except:
        pass
    
    try:
        os.remove("data.json")
    except:
        pass
    
    try:
        shutil.rmtree(directory)
    except Exception as e:
        print(e)
    
    try:
        os.remove("errorFile.txt")
    except:
        pass
    
    print(Fore.WHITE + "==============================\n\n\n")
    sys.exit()


#intro text
print("\n\n\n==============================")
print(Fore.YELLOW + "VIDEO COLLECTION")


try:
    #create file for C++ and python communication
    f = open("updateFile.txt", "w")
    f.close()
    
    f = open("errorFile.txt", "w")
    f.close()
except:
    error_handler("Unable to create updateFile.txt", directory = "")



try:
    #read JSON and get value file_name
    with open('data.json', 'r') as myfile:
        data = myfile.read()
    obj = json.loads(data)
    dir_name = obj["file_naming"]
    myfile.close()
except:
    error_handler("Unable to pares data.json file", dir_name)



try:
    #create a dynamic directory based on user inputs from frontend
    os.mkdir(str(dir_name))
    path = str(dir_name) + "/" + str(dir_name)

    print(Fore.YELLOW + "Created dynamically named directory")
    
    shutil.copy("./data.json", "./" + path + "_settings.json")
except: 
    error_handler("Unable to create dynamically named directory", dir_name)



cap = cv2.VideoCapture(1) #1 for connection to the video capture card



#A check to make sure that the Video is Displaying Properly
print(Fore.YELLOW + "Press SPACE to confirm what is being recorded. If the screen is not being displayed properly, hold SHIFT + N")
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret == True: #making sure capture was succesful
        cv2.imshow("Output", frame)
        if cv2.waitKey(100) & 0xFF == ord(" "):  #Stops when space is pressed to move onto recording
            break
        
        elif cv2.waitKey(100) & 0xFF == ord("N"):  #Stops when space is pressed to move onto recording
            error_handler("You exited because the incorrect video was beinig displayed, please check that everything is plugged in properly", dir_name)
    
    else:
        error_handler("Unable to connect to video card", dir_name)



try: 
    #setting up system to save video
    fourcc = cv2.VideoWriter_fourcc(* "mp4v")
    out = cv2.VideoWriter(str(dir_name) + "/" + str(dir_name) + ".mp4", fourcc, 30.0, (640, 480))
except:
    error_handler("Unable to set up video capture system", dir_name)


try: 
    #starts C++ file for kinematic data collection
    os.startfile("main.exe")  
except: 
    error_handler("Unable to start kinematic data collection", dir_name)



#Actually starting video recording
print(Fore.YELLOW + "Video recording started")
i = 0


while cap.isOpened():
    ret, frame = cap.read()
    
    if i == 0:
        start_time_recording = round(time.time() * 1000)
        i+= 1
        
    
    if ret == True: #making sure capture was succesful
        out.write(frame)
        cv2.imshow("Output", frame)
        
        
        #Checking for when the kinematic data stops
        f = open("updateFile.txt", "r")
        data = f.read()
        f.close()
        if data == "end":
            os.remove("updateFile.txt")
            break
        
        #checking for errors in the kinematic data collection process
        g = open("errorFile.txt", "r")
        data = g.read()
        g.close()
        if data == "error":
            error_handler("Something went wrong in the kinematic data collection during recording", dir_name)
    
    else:
        error_handler("Unable to connect to video card", dir_name)

#shut down connection to capture card and errorFile (comm file not needed)
cap.release()
cv2.destroyAllWindows()
out.release()
os.remove("errorFile.txt")
print(Fore.YELLOW + "Video recording ended")


#start of sync
print(Fore.YELLOW + "Working to sync ...")



try: 
    #getting start time of kinematic data
    print(Fore.YELLOW + "Reading start of kinematic time ...")
    
    with open(path + ".csv", "r") as csvFile:
        first_finder = csv.reader(csvFile)
        header = next(first_finder) #ignore
        data = next(first_finder)
        start_time_kinematic = round(float(data[8]) * 1000)

    print("Kinematic start time: " + str(start_time_kinematic))
except:
    error_handler("Error reading start of kinematic data", dir_name)
    
    


#create the csv file for frame timestamps
print(Fore.YELLOW + "Creating time stamp csv file ...")
frame_csv_file = open(path + "_frames.csv", "w", newline='')
writer = csv.writer(frame_csv_file)
header = ["Frame Number", "Timestamp"]
writer.writerow(header)



#for optimizing start time and record keeping for one frame below to start csv data 
new_dif = 9999999999999999
new_frame_stamp = start_time_recording


#reopen the mp4 file to clip
print(Fore.YELLOW + "Opening up video for post processsing ...")
cap = cv2.VideoCapture(path + ".mp4")

#switch on when to start writing to csv file (based on start of sync)
start_csv_writing = False
frame_no = 0


while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()

    #case where no sync yet
    if frame_exists and not start_csv_writing:

        old_frame_stamp = new_frame_stamp
        new_frame_stamp = cap.get(cv2.CAP_PROP_POS_MSEC) + start_time_recording #get stamp
        old_dif = new_dif 
        new_dif = abs(new_frame_stamp - start_time_kinematic) #get dif in start of kinematic and stamp

        #when the differnce of the new frame is more than the old one, we have already gotten to the bottom
        if new_dif > old_dif:
            print("Difference is: " + str(old_dif) + " at frame: " + str(frame_no - 1))

            break_frame = frame_no - 1

            frame_data = frame_data = [str(frame_no - 1), str(old_frame_stamp)]
            writer.writerow(frame_data)

            frame_data = frame_data = [str(frame_no), str(new_frame_stamp)]
            writer.writerow(frame_data)

            start_csv_writing = True
    
    #case when sync'd
    elif frame_exists and start_csv_writing:

        new_frame_stamp = cap.get(cv2.CAP_PROP_POS_MSEC) + start_time_recording
        frame_data = [str(frame_no), str(new_frame_stamp)]
        writer.writerow(frame_data)

        
        #stops repeats at the end
        if new_frame_stamp == old_frame_stamp:
            break
        
        old_frame_stamp = new_frame_stamp

    #something went wrong
    else:
        break
        
    frame_no += 1

cap.release()
frame_csv_file.close()
print(Fore.WHITE + "Completed video sync")



def trim(input_path, output_path, start):
    
    cap = cv2.VideoCapture(input_path)
    end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    input_vid = ffmpeg.input(input_path)

    (
        input_vid
        .trim (start_frame = start, end_frame = end)
        .setpts ('PTS-STARTPTS')
        .output (output_path)
        .run()
    )
    
    cap.release()
    
print(Fore.WHITE + "Starting video trimming")

try:
    trim(path + ".mp4", path + "_video.mp4", break_frame)
    os.remove(path + ".mp4")
except:
    error_handler("Error trimming video file", dir_name)
print(Fore.WHITE + "Completed video trimming")


print(Fore.WHITE + "Starting data synch")

def get_num_sensors_and_write_first_datapoint(org_data_reader, synced_data_writer):
    
    sensor_no_storage = 0
    
    while(1): 
        data = next(org_data_reader)
        new_sensor_no = int(data[0])

        if new_sensor_no < sensor_no_storage:
            print("No sensors: " + str(sensor_no_storage + 1))
            return sensor_no_storage + 1

        sensor_no_storage = new_sensor_no
        synced_data_writer.writerow(data)
  
def find_frame_matched_timestamp_and_write(org_data_reader, synced_data_writer, frames_data_reader, num_sensors, first_time):
    
    if first_time:
        for i in range(num_sensors - 2):
            try:
                next(org_data_reader)
            except StopIteration:
                return 1
        
    try:
        frame_stamp = float(next(frames_data_reader)[1])
    except StopIteration:
        return 1

    old_diff = 999999999999
    old_data = None
    data_buff = []

    while(1):

        try:
            new_data = next(org_data_reader)
        except StopIteration:
            return 1

        data_stamp = float(new_data[8]) * 1000
        new_diff = abs(frame_stamp - data_stamp)

        if new_diff > old_diff:
            print("Frame stamp: " + str(frame_stamp))
            print("Data stamp : " + str(float(old_data[8])*1000))
            print("Difference : " + str(abs(float(old_data[8])*1000) - frame_stamp))
            
            for item in data_buff:
                synced_data_writer.writerow(item)
            synced_data_writer.writerow(old_data)

            return 0

        data_buff = []
        old_data = new_data
        old_diff = new_diff
        for i in range(num_sensors - 1):
            data_buff.append(next(org_data_reader))

with open(path + ".csv", "r") as org_data_file, open(path + "_sync.csv", "w", newline='') as synced_data_file, open(path + "_frames.csv", "r") as frames_data_file:
    
    #preparing csv files to be read or to be written to
    org_data_reader = csv.reader(org_data_file)
    frames_data_reader = csv.reader(frames_data_file)
    synced_data_writer = csv.writer(synced_data_file)

    #copying header info to synced data file
    header = next(org_data_reader)
    synced_data_writer.writerow(header)

    num_sensors = get_num_sensors_and_write_first_datapoint(org_data_reader, synced_data_writer)

    #skips past the frames header and first data point(already synced)
    next(frames_data_file)
    next(frames_data_file)

    first_time = True

    #loops through all the frames and finds the closest data point
    while(1):
        result = find_frame_matched_timestamp_and_write(org_data_reader, synced_data_writer, frames_data_reader, num_sensors, first_time)
        first_time = False
        if result == 1:
            print("Done")
            break
            
print(Fore.WHITE + "Completed data sync")
print(Fore.WHITE + "==============================\n\n\n")