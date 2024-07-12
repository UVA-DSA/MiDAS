import tkinter as tk
from time import time_ns, sleep
import serial
import csv
import queue
from config import PDS_ON_THRESHOLD, PDS_PORT, PDS_LONG_THRESHOLD
#Each letter corresopnds to a pedal output. Arduino will output pedal data as single string
#format if 'A123' meaning the first pedal was pressed at pressure of 123. 'F999 G200' means last two pedals pressed, second to last being pressed harder
letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G'}

#Loop that constantly looks for connection, is called whenever arduino is not connected
def serialConnect(thread_stop):
    global arduino
    while True:
        if thread_stop.is_set():
            print("[PDS: Thread stop set, exiting..]")
            arduino = None
            return  
        try:
            arduino = serial.Serial(PDS_PORT, baudrate=9600, timeout=.1)
            return
        except serial.SerialException:
            print("[PDS: Serial Not Found, Retrying in 5 Seconds..]")
            sleep(5)

def get_PDS_data(q,path,thread_stop):
    global arduino
    # Create a CSV file and write the header row
    csv_path = path + '/PDS.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer_PDS = csv.writer(csv_file)
    csv_writer_PDS.writerow(['Pedal 1 Pressure', 'Pedal 2 Pressure','Pedal 3 Pressure','Pedal 4 Pressure','Pedal 6 Pressure', 'Pedal 7 Pressure','Pedal 1 Pressed','Pedal 2 Pressed','Pedal 3 Pressed','Pedal 4 Pressed','Pedal 5 Pressesd','Pedal 6 Pressed', 'Pedal 7 Pressed', 'Computer Time'])
    #Main loop that constantly collects serial data and sends it through the PDS queue, sends -1's for any pedal not presssed
    while True:
      
        serialConnect(thread_stop)
        
        if(arduino == None): # thread_stop is set
            break

        
        while arduino.is_open:
            
            if thread_stop.is_set():
                print("[PDS: Thread stop set, exiting..]")
                arduino.close() 
                break  
            try:
                out = arduino.readline().decode().strip()
                ret = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                #print(out, end = '')
                for let in letters:
                    num = ord(let) - ord('A')
                    if (out.find(let) != -1):
                        temp = float((out[(1 + out.find(let)):(8 + out.find(let))]).strip())
                    else:
                        temp = -1
                    ret[num] = temp
                    if ord(let) != ord('G'):
                        if temp >= PDS_ON_THRESHOLD:
                            ret[num+7] = 1
                        else:
                            ret[num+7] = 0
                    else:
                        if temp >= PDS_LONG_THRESHOLD:
                            ret[num+7] = 1
                        else:
                            ret[num+7] = 0
                #inserts the current time at the end of the data
                ret.insert(14,time_ns())
                try:
                    q.put(ret, block=False)
                    # print(ret)
                except:
                    while not q.empty():
                        q.get()
                csv_writer_PDS.writerow(ret)
                csv_file.flush()
            except KeyboardInterrupt:
                print("[PDS: KeyboardInterrupt: Exiting...]")
                arduino.close()
                csv_file.close()
                return
            except serial.SerialException:
                break