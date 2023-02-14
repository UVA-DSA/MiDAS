import socket
import time
import pickle

def send_num_incrementing():
    time.sleep(5)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 12346))
        num = 0
        while True:
            data = [num, num]
            s.sendall(pickle.dumps(data))
            num -= 1
            time.sleep(1)
    return

if __name__ == '__main__':
    send_num_incrementing()