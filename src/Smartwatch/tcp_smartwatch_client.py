import socket
import os
import csv
import time
from datetime import datetime
from multiprocessing import Queue
from typing import List
from threading import Event

def receive_smartwatch_data(server_ip: str, server_port: int, fifo_queue: Queue, recording_dir: str, smartwatch_id: str, thread_stop) -> None:
    try:
        # Create the necessary directories and CSV file for recording data
        columns: List[str] = ['sw_epoch_ms', 'wrist_position', 'sensor_type', 'value_X_Axis', 'value_Y_Axis', 'value_Z_Axis', 'seq_num' 'server_epoch_ms']
        curr_date = datetime.now()
        dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")
        newpath = os.path.join(recording_dir, f"smartwatch_data/sw_{smartwatch_id}/")
        
        os.makedirs(newpath, exist_ok=True)
        
        csv_file_path = os.path.join(newpath, 'sw_data.csv')
        
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            
            while True:
                client_socket = None
                connected = False
 
                while not connected:
                    if thread_stop.is_set():
                        # print("[Smartwatch: Thread stop set, exiting..]")
                        break
                    try:
                        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client_socket.settimeout(5)
                        print(f"[Smartwatch: Attempting to connect to server {server_ip}:{server_port}]")
                        client_socket.connect((server_ip, server_port))
                        connected = True
                        print(f"[Smartwatch: Successfully connected to server {server_ip}:{server_port}")
                    except Exception as e:
                        print(f"[Smartwatch: Connection failed: {e}. Retrying in 5 seconds...]")
                        time.sleep(5)
                
                message = "Hello, Smart Watch!"
                
                if thread_stop.is_set():
                    print("[Smartwatch: Thread stop set, exiting..]")
                    break
                
                try:
                    while True:
                        try:

                            if thread_stop.is_set():
                                print("[Smartwatch: Thread stop set, exiting..]")
                                break
                
                            # Send the message
                            client_socket.sendall(message.encode('utf-8'))

                            # Receive response from the server
                            # print("I am here")
                            response = client_socket.recv(1024)
                            if not response:
                                raise ConnectionError("[Smartwatch: Server closed the connection.]")
                            
                            sw_data = response.decode('utf-8').split(',')

                            curr_epoch_time = int(time.time_ns())

                            # Convert to proper types
                            sw_data[0] = int(sw_data[0])
                            sw_data[3] = float(sw_data[3])
                            sw_data[4] = float(sw_data[4])
                            sw_data[5] = float(sw_data[5])
                            sw_data[7] = float(sw_data[7])
                            sw_data.append(curr_epoch_time)
                            
                            writer.writerow(sw_data)
                            
                            
                            try:
                                # Add it to the queue to be processed by the main process
                                fifo_queue.put(sw_data, block=False)
                            except Exception as e:
                                # print("Error when writing to the FIFO: Clearing the queue ",e)
                                
                                while not fifo_queue.empty():
                                    # print("Queue size: ", fifo_queue.qsize())
                                    fifo_queue.get()
                        
                        except Exception as e:
                            print(f"[Error occurred while communicating with server: {e}]")
                            break  # Exit the inner loop and attempt to reconnect

                        except KeyboardInterrupt:
                            print("[Smartwatch receival interrupted by user. Exiting...]")
                            break

                except Exception as e:
                    print(f"[Smartwatch: Error: {e}]")
                
                except KeyboardInterrupt:
                    print("[Smartwatch: Smartwatch receival interrupted by user. Exiting...]")
                    break

                if client_socket:
                    client_socket.close()
                    print("[Smartwatch: Smartwatch connection closed!]")

    except KeyboardInterrupt:
        print("[Smartwatch: Smartwatch receival interrupted by user. Exiting...]")
        return


# smartwatch_1_ip = '172.27.191.158'
# smartwatch_port = 7889
# smartwatch_1_id = 'right'

# smartwatch_2_ip = '192.168.0.12'
# smartwatch_2_id = 'left'

# smartwatch_1_q = Queue()
# smartwatch_2_q = Queue()

# thread_stop = Event()
# # Call the function to send the message
# receive_smartwatch_data(smartwatch_2_ip, smartwatch_port, smartwatch_2_q, './test/', smartwatch_2_id, thread_stop)
