import pandas as pd
import magpylib as magpy
import pyquaternion
import numpy as np
from magpylib.source.magnet import Box
from magpylib import source, Collection

#Magnetization and dimension of Box magnet used
mag = [-575.4,0,0]
dim = [6.35,6.35,6.35]

#File read for data handling
data = pd.read_csv('data/rope_005.csv')        
print(data['joint_index'])
#x = list(data['z'])
#z = -1*np.array(list(data['x']))

#data['x'] = x
#data['z'] = z

#data = data.drop(data[data['joint_index']==0].index)
#data = data.reset_index()

#Initialize new lists for dataframe
angle = []
axis = []
for i in range(len(data['time'])):
    #Convert quaternions to angle axis
    quat = pyquaternion.Quaternion(a=data['wr'][i],b=data['xr'][i],c=data['yr'][i],d=data['zr'][i])
    angle.append(np.degrees(quat.angle))
    axis.append(quat.axis)

#add angle axis to panda
data['angle'] = angle
data['axis'] = axis

#determine amount of joints
joints = max(data['joint_index'])

#Determine amount of timesteps in system
timesteps = data.time.unique()

#All sensor readings for each timestep
sensor_readings = []

for j in range(len(timesteps)):
    #Get all data for timestep
    timestep_data = data[data['time'].isin([timesteps[j]])]
    
    #Separate sensors and magnets for timestep
    sensors = timestep_data[timestep_data['joint_index'].isin(np.arange(0,joints,2))]
    magnets = timestep_data[timestep_data['joint_index'].isin(np.arange(1,joints+1,2))]
   
    b = []
    for i in range(len(magnets['angle'])):
        if(np.array_equal(list(magnets['axis'])[i],[0.0,0.0,0.0])):
            b.append(Box(mag=mag,dim=dim,pos=[list(magnets['x'])[i],list(magnets['y'])[i],list(magnets['z'])[i]]))
        else:
            b.append(Box(mag=mag,dim=dim,pos=[list(magnets['x'])[i],list(magnets['y'])[i],list(magnets['z'])[i]],angle=list(magnets['angle'])[i],axis=list(magnets['axis'])[i]))
    col = Collection(b)
    
    s = []
    for i in range(len(sensors['angle'])):
        if(np.array_equal(list(sensors['axis'])[i],[0.0,0.0,0.0])):
            sensor = magpy.Sensor(pos=[list(sensors['x'])[i],list(sensors['y'])[i],list(sensors['z'])[i]])
        else:
            sensor = magpy.Sensor(pos=[list(sensors['x'])[i],list(sensors['y'])[i],list(sensors['z'])[i]],axis=list(sensors['axis'])[i],angle=list(sensors['angle'])[i])
        s.append(sensor)

        reading = np.round(sensor.getB(col),4)
        sensor_readings.append(reading)
        sensor_readings.append(reading)
    
    #print(i)
    #magpy.displaySystem(col,sensors=s)
sensor_readings = np.array(sensor_readings)
data['sensor_x'] = sensor_readings[:,0]
data['sensor_y'] = sensor_readings[:,1]
data['sensor_z'] = sensor_readings[:,2]
#xynoise = np.random.normal(100,.233,(len(data['sensor_data']),2))
#znoise = np.random.normal(100,.2,len(data['sensor_data']))

#noise = []
#for i in range(len(data['sensor_data'])):
#    noise.append(np.append(xynoise[i],znoise[i]))
#noise = np.array(noise)

#noisy_sensor_data = ((noise*.01)*np.array(list(data['sensor_data'])))
#data['noisy_sensor'] = list(noisy_sensor_data)
data['axis_x'] = [item[0] for item in data['axis']]
data['axis_y'] = [item[1] for item in data['axis']]
data['axis_z'] = [item[2] for item in data['axis']]

data.to_csv('data/processed.csv')
