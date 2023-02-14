import socket
import subprocess
import pickle

def receive_num():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
        s1.bind(('localhost', 12345))
        s1.listen()
        

        s2.bind(('localhost', 12346))
        s2.listen()

        conn1, _ = s1.accept()
        print("Connected to test2")
        conn2, _ = s2.accept()
        print("Connected to test3")
        while True:
            data1 = pickle.loads(conn1.recv(4096))
            if not data1:
                break
            data2 = pickle.loads(conn2.recv(4096))
            if not data2:
                break
            print(data1)
            print(data2)

if __name__ == '__main__':
    subprocess.Popen(['python3', '/Users/shrisha/Desktop/Research/DataCollectionSystem/test/test2.py'])
    
    subprocess.Popen(['python3', '/Users/shrisha/Desktop/Research/DataCollectionSystem/test/test3.py'])
    
    receive_num()