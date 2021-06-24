import serial
import numpy as np
import time
import pandas as pd

ser = serial.Serial('/dev/ttyACM0')
time.sleep(6)
total_readings = []
init_time = time.time()
Name = "90deg_3"
print(Name)
while(time.time() - init_time < 10):
    serial_data = ser.readline()
    serial_data = str(serial_data)[2:-5].split(',')
    try:
        serial_data = list(map(float,serial_data))
        temp = serial_data[0]
        serial_data[0] = serial_data[2]
        serial_data[2] = temp
    except:
        serial_data = [0,0,0]
    if len(serial_data) !=3:
        serial_data = [0,0,0]

    serial_data = np.array(serial_data)/1000
    
    if(serial_data[0] == 0 and serial_data[1] == 0 and serial_data[2] == 0):
        total_readings = []
    else:
        total_readings.append(serial_data)
        #print(serial_data)

total_readings = total_readings[1:]
total_readings = np.array(total_readings)
print(total_readings)

print("Num of Readings")
print(len(total_readings))
pd.DataFrame(total_readings).to_csv("calibrate_data/pidir/" + Name + ".csv",header=None,index=None)

print("Averages:")
print("X value:",np.mean(total_readings[:,0]))
print("Y value:",np.mean(total_readings[:,1]))
print("Z value:",np.mean(total_readings[:,2]))
