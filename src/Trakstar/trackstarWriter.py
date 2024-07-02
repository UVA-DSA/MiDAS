import os
import csv
import socket
from queue import Queue
from time import time_ns
import time
import multiprocessing
import subprocess



def get_trakstar_data(q, path):
    
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('127.0.0.1', 12346)
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()
    print(f'connected! with client address {client_address}')
    


    # Create a CSV file and write the header row
    csv_path = os.path.join(path, '/trakstar.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # So the line below is asking to append 10 items per line to the CSV but 9 values are being sent
    csv_writer.writerow(['SensorID', 'x', 'y', 'z', 'azimuth', 'elevation', 'roll', 'TrakStar Time', 'Computer Time'])

    while True:
        
        try:
            # Receive the data and split by commas
            data = connection.recv(128).decode()
            
            values = data.split(',')
            
            #convert to correct type
            #use this to get rid of erraneous values?
            # print("raw values",values)
            if(values[1]):
                values[1] = int(float(values[1]))
                for i in range(2, 9):
                    if(values[i]):
                        values[i] = float(values[i])
                
                #add in alienware time
                values.append(time_ns()) # TODO: replace with the actual trakstar time
                start_time = time.time()
                local_time = time.ctime(start_time)
                values.append(local_time)
                # print(values)
                
                #is this just writing all the values at once or each set of sensor values?
                csv_writer.writerow(values)
                q.put(values)
                if q.full():
                    _ = q.get() #removes last object from q to keep only a certain amount
                          
            acknowledgment_message = "ACK"
            connection.send(acknowledgment_message.encode())

        except KeyboardInterrupt:
            # Clean up the connection
            print("closed!")
            connection.close()
            csv_file.close()
            exit(-1)
        
if __name__ == "__main__":
    q = Queue()
    get_trakstar_data(q, "./test.csv")