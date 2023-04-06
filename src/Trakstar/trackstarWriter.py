import csv
import socket
from queue import Queue
from time import time_ns


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


    # Create a CSV file and write the header row
    csv_file = open(path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['SensorID', 'Status', 'x', 'y', 'z', 'azimuth', 'elevation', 'roll', 'TrakStar Time', 'Quality', 'Computer Time'])

    while True:
        
        try:
            # Receive the data and split by commas
            data = connection.recv(4096).decode()
            values = data.split(',')
            values.append(time_ns())

            #come in as strings, convert to floats before write
            for val in values:
                val = float(val.strip()) 
            
            csv_writer.writerow(values)
            q.put(data)
            if q.full():
                _ = q.get() #removes last object from q to keep only a certain amount

        finally:
            # Clean up the connection
            connection.close()
            csv_file.close()