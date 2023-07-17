import csv
import socket
from queue import Queue
from time import time_ns
import multiprocessing
import subprocess



def getTrackStarData(q, path):
    
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
    csv_path = path + '/trakstar.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['SensorID', 'Status', 'x', 'y', 'z', 'azimuth', 'elevation', 'roll', 'TrakStar Time', 'Computer Time'])

    while True:
        
        try:
            # Receive the data and split by commas
            data = connection.recv(128).decode()
            values = data.split(',')
            
            #convert to correct type
            if(values[0]):
                values[0] = int(float(values[0]))
                for i in range(1, len(values)):
                    if(values[i]):
                        values[i] = float(values[i])
                
                #add in alienware time
                values.append(time_ns())
                print(values)
                
                
                csv_writer.writerow(values)
            # q.put(data)
            # if q.full():
            #     _ = q.get() #removes last object from q to keep only a certain amount

        except Exception as e:
            # Clean up the connection
            print("closed!")
            print(e)
            connection.close()
            csv_file.close()
            return
        
if __name__ == "__main__":
    q = Queue(16)
    getTrackStarData(q, "./test.csv")