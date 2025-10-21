import time
import serial

arduino = serial.Serial("COM4", baudrate=9600, timeout=.1)
file = open(".\src\PDS\out.txt", "a")
letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G'}

while True:
    # file.write(str(round((1000 * time.time()))) + "\t" + out.strip() +"\n")
    out = arduino.readline().decode().strip()
    ret = [-1,-1,-1,-1,-1,-1,-1]
    #print(out, end = '')
    for let in letters:
        num = ord(let) - ord('A')
        if (out.find(let) != -1):
            temp = float((out[(1 + out.find(let)):(5 + out.find(let))]).strip())
        else:
            temp = -1
        ret[num]= temp
    print(ret)    