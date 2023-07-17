
import os

data = {'Task': 'Kay', 'Subject': 'S1', 'Task' : 'PT', 'Trial' : '01', 'Date' : '07_07_23'}

path = f"{data['Task']}_S{data['Subject']}_T{data['Trial']}_{data['Date']}/" 

mydir = "./Data/"
myfile = path
path = os.path.join(mydir, myfile)


if not os.path.exists(path):
    os.makedirs(path)