import tkinter as tk
from time import time_ns
import serial
import threading
import csv

def get_PDS_data(q,path):
    # Create a CSV file and write the header row
    csv_path = path + '/PDS.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Pedals Pressed','Computer Time'])

    arduino = serial.Serial("COM3", baudrate=9600, timeout=.1)

    while arduino.is_open:
        try:
            out = arduino.readline().decode()
            print(out, end = '')
            time = time_ns()
            csv_writer.writerow([out.strip(), time])
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting...")
            arduino.release()
            csv_file.close()
            exit(0)