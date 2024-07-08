import os
import csv
import socket
from queue import Queue
from time import time_ns
import time
import multiprocessing
import subprocess

def get_trakstar_data(q, path, thread_stop):
    
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 12346)
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)
    sock.settimeout(5)  # Set timeout to 5 seconds

    trakstar_dir = path + 'Trakstar'
    if not os.path.exists(trakstar_dir):
        os.mkdir(trakstar_dir)
    
    # Create a CSV file and write the header row
    csv_path = trakstar_dir + '/trakstar.csv'

    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Bytes','SensorID', 'x', 'y', 'z', 'azimuth', 'elevation', 'roll', 'TrakStar Time', 'Bytes','SensorID', 'x', 'y', 'z', 'azimuth', 'elevation', 'roll', 'TrakStar Time', 'Bytes','SensorID', 'x', 'y', 'z', 'azimuth', 'elevation', 'roll', 'TrakStar Time', 'Bytes','SensorID', 'x', 'y', 'z', 'azimuth', 'elevation', 'roll', 'TrakStar Time', 'Computer Time'])
    connect = -1
    out = []
    while True:
        if thread_stop.is_set():
            print("[TrakStar: Thread stop set, exiting..]")
            if 'connection' in locals():
                connection.close()
            csv_file.close()
            break
        
        try:
            print('[TrakStar: waiting for a connection]')
            connection, client_address = sock.accept()
            print(f'[TrakStar: connected! with client address {client_address}]')

            while True:
                if thread_stop.is_set():
                    print("[TrakStar: Thread stop set, exiting..]")
                    connection.close()
                    csv_file.close()
                    return
                
                try:
                    # Receive the data and split by commas
                    data = connection.recv(128).decode()
                    values = data.split(',')
                    #Convert data
                    if values[1]:
                        values[1] = int(float(values[1]))
                        for i in range(2, 9):
                            if values[i]:
                                values[i] = float(values[i])
                    if(values[1] == 0):
                        connect = 4
                        out = []

                    if(connect > 0):
                        out = out + values
                        connect = connect - 1
            
                    # Add in alienware time
                    values.append(time_ns())  # TODO: replace with the actual trakstar time
                    start_time = time.time()
                    local_time = time.ctime(start_time)
                    values.append(local_time)
                    csv_writer.writerow(values)
                    csv_file.flush()
                    
                    if(connect == 0):   
                        out.append(time_ns())  # TODO: replace with the actual computer time
                        start_time = time.time()
                        local_time = time.ctime(start_time)
                        out.append(local_time)
                        
                        q.put(out)
                    if q.full():
                        _ = q.get()  # Removes last object from q to keep only a certain amount
                    acknowledgment_message = "ACK"
                    connection.send(acknowledgment_message.encode())
                        

                except (socket.error, KeyboardInterrupt):
                    # Clean up the connection
                    print("[TrakStar: Connectionclosed!]")
                    connection.close()
                    csv_file.close()
                    exit(-1)
                
        except socket.timeout:
            print("[TrakStar: Socket timeout, retrying...]")
            if thread_stop.is_set():
                print("[TrakStar: Thread stop set during timeout, exiting..]")
                break
        except KeyboardInterrupt:
            print("[TrakStar: Interrupted by user, exiting..]")
            break

# if __name__ == "__main__":
#      q = Queue()
#      get_trakstar_data(q, "./test.csv")
