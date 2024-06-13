import time
import serial

arduino = serial.Serial("COM3", baudrate=9600, timeout=.1)
file = open(".\src\PDS\out.txt", "a")
while True:
    out = arduino.readline().decode()
    print(out, end = '')
    file.write(str(round((1000 * time.time()))) + "\t" + out.strip() +"\n")