import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy as np
import pandas as pd
file1 = 'bend_4_try_2'

df = pd.read_csv('./final_data/' + file1 + '.csv')

bend_direction = df['bend_direction']
bend_dir = df['bend_dir']

bend_angle = df['bend_angle']
bend_ang = df['bend_ang']

bend_direction = np.array(bend_direction)
for i in range(300,len(bend_direction)-100):
    if(bend_direction[i]>0 or bend_direction[i] < -1.5):
        bend_direction[i] = bend_direction[i-1]

time = df['time']
time = np.array(time)
bend_dir = np.array(bend_dir)
bend_direction = ndimage.median_filter(bend_direction,size=100)[500:3150]
bend_angle = ndimage.median_filter(bend_angle,size=100)
for i in range(1,len(bend_angle)):
    if(abs(abs(bend_angle[i]) - abs(bend_angle[i-1])) > .06):
            bend_angle[i] = bend_angle[i-1]

plt.style.use('seaborn-paper')
plt.gcf().subplots_adjust(bottom=.15)
plt.gcf().subplots_adjust(left=.15)
plt.tick_params(axis='both',labelsize=14)
plt.title("Bend Direction Readings: Motion Capture & Sensor",fontsize=16)
plt.plot(time,bend_dir,label='Sensor Bend Direction')
plt.plot(np.arange(0,len(bend_direction)/120,(1/120)),bend_direction,label="Motion Capture Bend Angle")
plt.xlabel("Time (seconds)",fontsize=16)
plt.ylabel("Bend Direction (rad)",fontsize=16)
plt.legend(fontsize=11)
plt.show()
plt.gcf().subplots_adjust(bottom=.15)
plt.gcf().subplots_adjust(left=.15)
plt.tick_params(axis='both',labelsize=14)
plt.title("Bend Angle Readings: Motion Capture & Sensor",fontsize=16)
plt.plot(time,bend_ang,label='Sensor Bend Angle')
plt.plot(np.arange(0,len(bend_angle)/120,(1/120)),-(bend_angle)+np.pi/2,label='Motion Capture Bend Angle')
plt.xlabel("Time (seconds)",fontsize=16)
plt.ylabel("Bend Angle (rad)",fontsize=16)
plt.legend(fontsize=11)
plt.show()

