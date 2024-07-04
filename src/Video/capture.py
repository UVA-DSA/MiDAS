import cv2  
from queue import Queue
from time import time_ns
import csv
import os
from config import OBS_CAPTURE_PORT
from PIL import ImageTk,Image
import multiprocessing as mp


def save_images(q, path):
    image_sequence_num = 0

    # Create a CSV file and write the header row
    csv_path = path + '/video.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame #','Computer Time', 'Path'])

    while True:
        try:
            time, img_path, frame = q.get()
            print("Saving image ...", img_path)

            cv2.imwrite(img_path,frame)

            image_sequence_num += 1
            
            csv_writer.writerow([image_sequence_num, time, img_path])
            csv_file.flush()


        except Exception as e:
            print("Error while saving image frame..")

        except KeyboardInterrupt:
            print("Keyboard interrupt received..")
            break
    
    csv_file.close()


def send_vid_data(q, path, img_q):


    img_dir = path + 'imgs'
        
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    
    cap = cv2.VideoCapture(OBS_CAPTURE_PORT, apiPreference=cv2.CAP_ANY, params=[
    cv2.CAP_PROP_FRAME_WIDTH, 3840,
    cv2.CAP_PROP_FRAME_HEIGHT, 1080]) #2 for connection to the video capture card if using realsense too, set to 0 for testing with web cam
    frame_num = 0

    image_save_q = mp.Queue(maxsize=1024)
    # image_saving_process =  mp.Process(target=save_images, args=(image_save_q, img_dir))
    # image_saving_process.start()

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret == True: #making sure capture was succesful
                frame_num += 1
                time = time_ns()
                filename = f"/{frame_num}_{time}.png"
                img_path = img_dir + filename

                data = [ time, img_path]
                # image_save_data = [time, img_path, frame]
                image_save_data = [time, img_path]

                # print("Image save queue size: ",image_save_q.qsize())
                # try:
                #     image_save_q.put(image_save_data, block=False)
                # except:
                #     print("Image save Queue full!")
                    # while not image_save_q.empty():
                    #     image_save_q.get()
                # image compression and conversion to be sent to display
                # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # image = Image.fromarray(image)
                # image = ImageTk.PhotoImage(image)
                try:
                    img_q.put(frame)
                except:
                    while not img_q.empty():
                        q.get()
                try:
                    q.put(data)
                except:
                    while not q.empty():
                        q.get()

                # if q.full():
                #     _ = q.get() #removes last object from q to keep only a certain amount

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting...")
            cap.release()
            exit(0)