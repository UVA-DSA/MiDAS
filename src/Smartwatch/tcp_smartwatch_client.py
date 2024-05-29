import socket
import os
import csv
import time
from datetime import datetime



def receive_smartwatch_data(server_ip, server_port, fifo_queue, recording_dir):
    # Create a socket object


    try:
        # Connect to the server
        print(f"Connecting to server {server_ip}:{server_port}")
        
        client_socket = None
        connected = False
        while not connected:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                # Attempt to connect to the server
                print(f"Attempting to connect to server {server_ip}:{server_port}")
                client_socket.connect((server_ip, server_port))
                connected = True
                print(f"Successfully connected to server {server_ip}:{server_port}")
            except Exception as e:
                print(f"Connection failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        
        columns = ['sw_epoch_ms','wrist_position','sensor_type','value_X_Axis','value_Y_Axis','value_Z_Axis','server_epoch_ms']

        curr_date = datetime.now()
        dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

        newpath = f"{recording_dir}/smartwatch_data/"
        
        message = "Hello, Smart Watch!"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        with open(newpath+'sw_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
 
            while True:
                try:
                    # Send the message
                    client_socket.sendall(message.encode('utf-8'))

                    # Receive response from the server (optional)
                    response = client_socket.recv(1024)
                    
                    sw_data = response.decode('utf-8').split(',')

                    curr_epoch_time = int(time.time_ns())
                    
                    # convert to proper types
                    sw_data[0] = int (sw_data[0])
                    sw_data[3] = float (sw_data[3])
                    sw_data[4] = float (sw_data[4])
                    sw_data[5] = float (sw_data[5])
                    
                    sw_data.append(curr_epoch_time) #check

                    writer.writerow(sw_data)
                    
                    print(f"Received from server: {sw_data}")
                    
                    #add it to the queue to be processed by the main process
                    fifo_queue.put(sw_data) 
                 
            
                except Exception as e:
                    print("Error occured", e)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the socket
        client_socket.close()


# # Call the function to send the message
# send_message(server_ip, server_port, message_to_send)
