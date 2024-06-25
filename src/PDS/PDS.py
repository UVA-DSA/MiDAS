import tkinter as tk
from time import time_ns
import serial
import csv
import queue

def get_PDS_data(q,path):
    # Create a CSV file and write the header row
    csv_path = path + '/PDS.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer_PDS = csv.writer(csv_file)
    csv_writer_PDS.writerow(['Pedals Pressed','Computer Time'])
    
    while True:
        try:
            arduino = serial.Serial("COM3", baudrate=9600, timeout=.1)
            break
        except serial.SerialException:
            print("Serial Not Found")
            return

    while arduino.is_open:
        try:
            out = arduino.readline().decode()
            #print(out, end = '')
            q.put(out)
            time = time_ns()
            csv_writer_PDS.writerow([(out.strip()), str(time)])
            csv_file.flush()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting...")
            arduino.release()
            csv_file.close()
            return