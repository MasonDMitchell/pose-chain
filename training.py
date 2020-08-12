import pandas as pd
import numpy as np

def input_data(filename="data/processed.csv"):
    #Read data from file
    df = pd.read_csv(filename)

    #Determine # of joints, pairs & timesteps
    joint_num = max(df['joint_index'])+1
    timestep_num = len(set(df['time']))
    pair_num = joint_num//2
    
    #Extract pose data from file
    pos = df[['x','y','z']].to_numpy()
    angle = df['angle'].to_numpy()
    axis = df[['axis_x','axis_y','axis_z']].to_numpy()
    
    #Split pose data for magnets and sensors
    sensor_pos,sensor_angle,sensor_axis = pos[::2],angle[::2],axis[::2]
    magnet_pos,magnet_angle,magnet_axis = pos[1::2],angle[1::2],axis[1::2]

    #Extract sensor data & reshape for use
    sensor_data = df[['sensor_x','sensor_y','sensor_z']].to_numpy()[::2]

    return pair_num, sensor_data, sensor_pos, sensor_angle, sensor_axis, magnet_pos, magnet_angle, magnet_axis

if __name__ == "__main__":
    input_data()
