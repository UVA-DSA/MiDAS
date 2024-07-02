import socket
import os
import csv
import time
from datetime import datetime
from multiprocessing import Queue, Process
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def receive_smartwatch_data(server_ip: str, server_port: int, fifo_queue: Queue, recording_dir: str, smartwatch_id: str) -> None:
    try:
        # Create the necessary directories and CSV file for recording data
        columns: List[str] = ['sw_epoch_ms', 'wrist_position', 'sensor_type', 'value_X_Axis', 'value_Y_Axis', 'value_Z_Axis', 'server_epoch_ms']
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
                    try:
                        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        print(f"Attempting to connect to server {server_ip}:{server_port}")
                        client_socket.connect((server_ip, server_port))
                        connected = True
                        print(f"Successfully connected to server {server_ip}:{server_port}")
                    except Exception as e:
                        print(f"Connection failed: {e}. Retrying in 5 seconds...")
                        time.sleep(5)
                
                message = "Hello, Smart Watch!"
                
                try:
                    while True:
                        try:
                            # Send the message
                            client_socket.sendall(message.encode('utf-8'))

                            # Receive response from the server
                            response = client_socket.recv(1024)
                            if not response:
                                raise ConnectionError("Server closed the connection.")
                            
                            sw_data = response.decode('utf-8').split(',')

                            curr_epoch_time = int(time.time_ns())

                            # Convert to proper types
                            sw_data[0] = int(sw_data[0])
                            sw_data[3] = float(sw_data[3])
                            sw_data[4] = float(sw_data[4])
                            sw_data[5] = float(sw_data[5])
                            sw_data.append(curr_epoch_time)
                            
                            writer.writerow(sw_data)
                            
                            try:
                                # Add it to the queue to be processed by the main process
                                fifo_queue.put(sw_data, block=False)
                            except Exception as e:
                                print("Error when writing to the FIFO: Clearing the queue ", e)
                                while not fifo_queue.empty():
                                    fifo_queue.get()
                        
                        except Exception as e:
                            print(f"Error occurred while communicating with server: {e}")
                            break  # Exit the inner loop and attempt to reconnect

                except Exception as e:
                    print(f"Error: {e}")
                
                finally:
                    if client_socket:
                        client_socket.close()
                        print("Smartwatch connection closed!")
                        break

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        return

def plot_smartwatch_data(fifo_queue: Queue) -> None:
    # Initialize the plot
    fig, ax = plt.subplots()
    xdata, ydata_x, ydata_y, ydata_z = [], [], [], []
    ln_x, = plt.plot([], [], 'r-', label='X-Axis')
    ln_y, = plt.plot([], [], 'g-', label='Y-Axis')
    ln_z, = plt.plot([], [], 'b-', label='Z-Axis')

    def init():
        ax.set_xlim(0, 1000)
        ax.set_ylim(-10, 10)  # Adjust the Y-axis limits based on expected data range
        ax.legend()
        return ln_x, ln_y, ln_z

    def update(frame):
        while not fifo_queue.empty():
            data = fifo_queue.get()
            if len(xdata) > 1000:
                xdata.pop(0)
                ydata_x.pop(0)
                ydata_y.pop(0)
                ydata_z.pop(0)
            xdata.append(data[0])
            ydata_x.append(data[3])
            ydata_y.append(data[4])
            ydata_z.append(data[5])
        ln_x.set_data(np.arange(len(ydata_x)), ydata_x)
        ln_y.set_data(np.arange(len(ydata_y)), ydata_y)
        ln_z.set_data(np.arange(len(ydata_z)), ydata_z)
        return ln_x, ln_y, ln_z

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50)
    plt.show()

if __name__ == '__main__':
    smartwatch_ip = '172.27.176.73'
    smartwatch_port = 7889
    smartwatch_id = 'left'
    fifo_queue = Queue()

    # Start the data receiving process
    data_process = Process(target=receive_smartwatch_data, args=(smartwatch_ip, smartwatch_port, fifo_queue, './test/', smartwatch_id))
    data_process.start()

    # Start the plotting
    plot_smartwatch_data(fifo_queue)

    data_process.join()
