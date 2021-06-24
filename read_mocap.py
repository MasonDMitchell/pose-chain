import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
from scipy import signal


def mocap_read(filename):
    df = pd.read_csv("./mocap/" +filename+".csv",header=6)

    quaternions = np.array(list(zip(df['X.2'],df['Y.2'],df['Z.2'],df['W.1'])))

    rotations = list(map(R.from_quat,quaternions[:-1]))

    z_vector_list = []
    for rotation in rotations:
        z_vector_list.append(rotation.apply([0,0,1]))

    z_vector_list = np.array(z_vector_list)

    z_vector_list = np.where(np.reshape(z_vector_list[:,2],(-1,1)) > 0,z_vector_list,-z_vector_list)

    '''
    sos = signal.butter(3,.7,'low',analog=False,fs=120,output='sos')
    z_vector_list[:,0] = signal.sosfilt(sos,z_vector_list[:,0])
    z_vector_list[:,1] = signal.sosfilt(sos,z_vector_list[:,1])
    z_vector_list[:,2] = signal.sosfilt(sos,z_vector_list[:,2])
    '''

    z_vector_list = z_vector_list[300:]

    bend_direction = np.arctan2(z_vector_list[:,1],z_vector_list[:,0])
    bend_angle = np.arcsin(z_vector_list[:,2])
    return bend_direction,bend_angle

if __name__ == "__main__":
    df = pd.read_csv("./mocap/bend_2_try_2.csv",header=6)

    quaternions = np.array(list(zip(df['X.2'],df['Y.2'],df['Z.2'],df['W.1'])))

    rotations = list(map(R.from_quat,quaternions[:-1]))

    z_vector_list = []
    for rotation in rotations:
        z_vector_list.append(rotation.apply([0,0,1]))

    z_vector_list = np.array(z_vector_list)

    z_vector_list = np.where(np.reshape(z_vector_list[:,2],(-1,1)) > 0,z_vector_list,-z_vector_list)

    '''
    sos = signal.butter(3,.7,'low',analog=False,fs=120,output='sos')
    z_vector_list[:,0] = signal.sosfilt(sos,z_vector_list[:,0])
    z_vector_list[:,1] = signal.sosfilt(sos,z_vector_list[:,1])
    z_vector_list[:,2] = signal.sosfilt(sos,z_vector_list[:,2])
    '''

    z_vector_list = z_vector_list[300:]

    bend_direction = np.arctan2(z_vector_list[:,1],z_vector_list[:,0])
    bend_angle = np.arcsin(z_vector_list[:,2])
    plt.plot((-bend_angle)+np.pi/2)
    plt.show()
    bend_direction = bend_direction
    for i in range(len(bend_direction)):
        if(bend_direction[i] < -np.pi):
            bend_direction[i] = bend_direction[i]+(np.pi*2)
    plt.plot(bend_direction)
    plt.show()
