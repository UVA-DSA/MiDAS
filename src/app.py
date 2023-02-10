import eel
import random
import datetime


eel.init('web')

@eel.expose
def sendData(subject, trial, task, rate, range_val, hemisphere, filter_val):
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
    return

eel.start('index.html')