import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from straight_filter import Filter
from tools import createChain,noise

def sensor_read(filename):
    df = pd.read_csv('./experiment_data/' + filename + '.csv',header=None)

    df = df[1:]
    sensor_data = df[0].to_numpy()
    time_data = df[1].to_numpy()
    real_data = []

    for data in sensor_data:
        data = data[1:-1].strip()
        real_data.append(np.fromstring(data,dtype=float,sep=' '))


    real_data = np.array(real_data)
    return real_data,time_data


def sensor_run(filename):
    df = pd.read_csv('./experiment_data/' + filename + '.csv',header=None)
    df = df[1:]
    sensor_data = df[0].to_numpy()
    time_data = df[1].to_numpy()
    real_data = []

    for data in sensor_data:
        data = data[1:-1].strip()
        real_data.append(np.fromstring(data,dtype=float,sep=' '))


    real_data = np.array(real_data)
    all_time_data = []
    all_bend_angle = []
    all_bend_direction = []
    
    cutoff = 10000
    time_data = time_data[:cutoff]
    real_data = real_data[:cutoff]
    
    chain = createChain(1000,1,0,0,13,0)
    x = Filter(chain,noise)

    avg_y = np.mean(real_data[:,1][:100])
    avg_z = np.mean(real_data[:,2][:100])
    #avg_y = -.05
    #ang_z = .116
    for serial_data in real_data:

        #serial_data[0] = serial_data[0] * 3.88296
        #serial_data[1] = (5.2799/(1+np.exp(-2.05192*(serial_data[1]-avg_y)))) - (5.2799/2)
        #serial_data[2] = (4.51854/(1+np.exp(-4.07619*(serial_data[2]-avg_z)))) - (4.51854/2)
        #serial_data[2] = 2.61751*serial_data[2] -.0322
        serial_data[0] = serial_data[0] * 10
        serial_data[1] = (serial_data[1]-avg_y) * 10
        serial_data[2] = (serial_data[2]-avg_z) * 10
        
        all_time_data.append(serial_data)

        x.sensor_data = serial_data
        x.compute_flux()
        x.reweigh()
        x.resample()
        pos,bend_a,bend_d = x.predict()
        #print("Bend Angle: {:<5f} Bend Direction: {:<5f}".format(bend_a,bend_d))
        x.update()

        all_bend_angle.append(bend_a)
        all_bend_direction.append(bend_d)

    all_time_data = np.array(all_time_data)
    
    #plt.plot(all_time_data[:,1],color='black')
    #plt.plot(all_time_data[:,2],color='red')
    #plt.show()
    plt.style.use('ggplot')
    plt.title("Bend Direction Reading")
    plt.ylabel("Bend Direction (rad)")
    plt.xlabel("Time (seconds)")
    plt.plot(time_data - time_data[0],all_bend_direction)
    plt.show()
    plt.style.use('ggplot')
    plt.title("Bend Angle Reading")
    plt.ylabel("Bend Angle (rad)")
    plt.xlabel("Time (seconds)")
    plt.plot(time_data - time_data[0],all_bend_angle)
    plt.show()
    
    return all_bend_direction, all_bend_angle,time_data

