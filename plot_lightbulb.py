from read_sensor import sensor_read, sensor_run
import numpy as np
import matplotlib.pyplot as plt

file1 = 'light_test_2'

bend_direction, bend_angle, time = sensor_run(file1)

for i in range(len(bend_direction)):
    if(bend_direction[i] <0):
        bend_direction[i] = bend_direction[i]+np.pi
    else:
        bend_direction[i] = bend_direction[i]-np.pi


plt.style.use('seaborn-paper')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.title("Bend Direction Reading",fontsize=20)
plt.xlabel("Time (seconds)",fontsize=16)
plt.ylabel("Bend Direction (rad)",fontsize=16)
plt.plot(time-time[0],bend_direction)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.tick_params(axis='both',which='minor',labelsize=14)
plt.show()

plt.style.use('seaborn-paper')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.title("Bend Angle Reading",fontsize=20)
plt.xlabel("Time (seconds)",fontsize=16)
plt.ylabel("Bend Angle (rad)",fontsize=16)
plt.plot(time-time[0],bend_angle)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.tick_params(axis='both',which='minor',labelsize=14)
plt.show()

