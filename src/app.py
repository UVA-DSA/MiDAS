import eel


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
    getInputStreams(data)
    return

def getInputStreams(data: dict):
    '''
    Uses data to create dynamic directory, and reach out to different input streams. 

    data: Dictionary that contains keys Subject, Trial, Task, Rate, Range, Hemisphere, and Filter.

    returns: Nothing
    '''
    return

eel.start('index.html')