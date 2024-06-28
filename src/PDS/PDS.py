import tkinter as tk
from time import time_ns
import serial
import csv
import queue

#Each letter corresopnds to a pedal output. Arduino will output pedal data as single string
#format if 'A123' meaning the first pedal was pressed at pressure of 123. 'F999 G200' means last two pedals pressed, second to last being pressed harder
letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G'}

#Loop that constantly looks for connection, is called whenever arduino is not connected
def serialConnect():
    global arduino
    while True:
        try:
            arduino = serial.Serial("COM3", baudrate=9600, timeout=.1)
            return
        except serial.SerialException:
            print("Serial Not Found")

def get_PDS_data(q,path):
    global arduino
    # Create a CSV file and write the header row
    csv_path = path + '/PDS.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer_PDS = csv.writer(csv_file)
    csv_writer_PDS.writerow(['Pedals Pressed', 'Computer Time'])
    #Main loop that constantly collects serial data and sends it through the PDS queue, sends -1's for any pedal not presssed
    while True:
        serialConnect()
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
                #inserts the current time at the end of the data
                ret.insert(7,time_ns())
                q.put(ret)
                csv_writer_PDS.writerow(ret)
                csv_file.flush()
            except KeyboardInterrupt:
                print("KeyboardInterrupt: Exiting...")
                arduino.close()
                csv_file.close()
                return
            except serial.SerialException:
                break