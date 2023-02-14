import eel
import socket
import subprocess
import pickle

#defines
video_port = 12345
video_stream_size = 4096

eel.init('web')

@eel.expose
def sendData(subject, trial, task, rate, range_val, hemisphere, filter_val):
    '''
    Takes data in from frontend, puts it in a dictionary, and passes dictionary into function 
    getInputStreams(). All data values come in as strings and are converted into respective 
    types.

    subject: Name of subject running trial
    trial: Trial number as an integer
    task: Name of task being performed
    rate: Rate in Hz as a float
    range_val: Range between 36, 72, and 144 in
    hemisphere: front, back, left, right, top, or bottom
    filter_val: wide or narrow.

    returns: Nothing
    '''
    data = {
        "Subject" : subject,
        "Trial": int(trial),
        "Task": task,
        "Rate": float(rate),
        "Range": float(range_val),
        "Hemisphere": hemisphere,
        "Filter": filter_val
        }
    print(data)
    path = "FILL OUT" #TODO
    getInputStreams(path)
    return

def getInputStreams(path: str):
    '''
    Reaches out to different data streams (socket) to get data. Prepares to write them to file 

    path: string that shows path where data should be stored

    returns: Nothing
    '''

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as video:
        video.bind(('localhost', video_port))
        video.listen()
        

        vid_con, _ = video.accept()
        print("Connected to Video")

        while True:

            #video data processing
            vid_data = pickle.loads(vid_con.recv(video_stream_size))
            if vid_data == ["Data not collected", "N/A"]: #something wrong with video data
                pass #update later with what to do


            if not vid_data:
                pass
            print(vid_data)
                
    return

eel.start('index.html')