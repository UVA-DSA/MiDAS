import socket
import time
import random

def simulate_smartwatch_data(server_ip, server_port):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Bind the socket to the address and port
        server_socket.bind((server_ip, server_port))
        server_socket.listen(1)
        print(f"Server started at {server_ip}:{server_port}, waiting for connection...")
        
        # Wait for a connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established.")
        
        sequence_number = 0
        
        while True:
            try:
                # Receive a message from the client
                message = client_socket.recv(1024)
                
                if not message:
                    break
                
                print(f"Received from client: {message.decode('utf-8')}")
                
                # Generate mock data
                sw_epoch_ms = int(time.time() * 1000)
                wrist_position = random.choice(['up', 'down', 'neutral'])
                sensor_type = random.choice(['accelerometer', 'gyroscope'])
                value_X_Axis = round(random.uniform(-10.0, 10.0), 2)
                value_Y_Axis = round(random.uniform(-10.0, 10.0), 2)
                value_Z_Axis = round(random.uniform(-10.0, 10.0), 2)
                
                #replaced sw_epoch_ms with sequence_number
                sw_data = f"{sequence_number},{wrist_position},{sensor_type},{value_X_Axis},{value_Y_Axis},{value_Z_Axis}"
                
                sequence_number += 1
                # Send the mock data to the client
                client_socket.sendall(sw_data.encode('utf-8'))
                
                print(f"Sent to client: {sw_data}")
                
                # Sleep for a short interval to simulate real-time data transmission
                # busy wait loop for 33 ms
                start_time = time.time()
                while (time.time() - start_time) < 0.033:
                    pass
            
            except Exception as e:
                print(f"Error occurred: {e}")
                break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Close the sockets
        client_socket.close()
        server_socket.close()

if __name__ == "__main__":
    SERVER_IP = "127.0.0.1"  # Localhost for testing
    SERVER_PORT = 7889      # Arbitrary non-privileged port
    
    simulate_smartwatch_data(SERVER_IP, SERVER_PORT)
