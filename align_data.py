from read_mocap import mocap_read
from read_sensor import sensor_read,sensor_run
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
file1 = "bend_4_try_2"

bend_direction,bend_angle = mocap_read(file1)
mocutoff1=500
mocutoff2=-301
bend_direction = bend_direction[mocutoff1:mocutoff2]
bend_angle = bend_angle[mocutoff1:mocutoff2]
df = pd.read_csv('./filter_readings/' + file1 + '.csv')
bend_dir = df['bend_dir'].to_numpy()
bend_ang = df['bend_ang'].to_numpy()
time = df['time'].to_numpy()

cutoff1 = 0
cutoff2 =22100
offset1 = 0
offset2 = 0
plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 42})

plt.title("Bend Direction Readings: Motion Capture & Sensor",fontsize=14)
plt.plot(time[cutoff1:cutoff2]-time[cutoff1],-(bend_dir[cutoff1:cutoff2]-offset1),label='Sensor Bend Direction')
plt.plot(np.arange(0,len(bend_direction)/120,(1/120)),bend_direction,label="Motion Capture Bend Angle")
plt.xlabel("Time (seconds)",fontsize=12)
plt.ylabel("Bend Direction (rad)",fontsize=12)
plt.legend(fontsize=12)
plt.show()
plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 42})
plt.title("Bend Angle Readings: Motion Capture & Sensor")
plt.plot(time[cutoff1:cutoff2]-time[cutoff1],bend_ang[cutoff1:cutoff2]-offset1,label='Sensor Bend Angle')
plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 42})
plt.plot(np.arange(0,len(bend_angle)/120,(1/120)),-(bend_angle)+np.pi/2,label='Motion Capture Bend Angle')
plt.xlabel("Time (seconds)")
plt.ylabel("Bend Angle (rad)")
plt.legend()
plt.show()

df = pd.DataFrame(bend_direction,columns=['bend_direction'])
df['bend_dir'] = pd.Series(bend_dir[cutoff1:cutoff2])
df['bend_ang'] = pd.Series(bend_ang[cutoff1:cutoff2])
df['time'] = pd.Series(time[cutoff1:cutoff2] - time[cutoff1])
df['bend_angle'] = bend_angle


df.to_csv('./final_data/' + file1 + '.csv')

