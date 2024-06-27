import tkinter as tk
from time import time_ns
import serial
import csv
import queue

letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G'}

def get_PDS_data(q,path):
    # Create a CSV file and write the header row
    csv_path = path + '/PDS.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer_PDS = csv.writer(csv_file)
    csv_writer_PDS.writerow(['Pedals Pressed', 'Computer Time'])
    
    while True:
        try:
            arduino = serial.Serial("COM3", baudrate=9600, timeout=.1)
            break
        except serial.SerialException:
            print("Serial Not Found")
            return

    while arduino.is_open:
        try:
            out = arduino.readline().decode().strip()
            ret = [-1,-1,-1,-1,-1,-1,-1]
            #print(out, end = '')
            for let in letters:
                num = ord(let) - ord('A')
                if (out.find(let) != -1):
                    temp = int((out[(1 + out.find(let)):(5 + out.find(let))]).strip())
                else:
                    temp = -1
                ret[num] = temp
            ret.insert(7,time_ns())
            q.put(ret)
            csv_writer_PDS.writerow(ret)
            csv_file.flush()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting...")
            arduino.close()
            csv_file.close()
            return