import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sci
from tools import bend,angle

#Creating pi to reduce confusion
pi = np.pi

segment_length = 300
segment_amount = 4
#set bend angles for each segment. List length should be same as segment amount.
bend_angle = [pi/2,0,0,0]
#set bend direction for each segment. List length should be same as segment amonut.
bend_direction = [pi/2,0,0,0]

#setting up graphing
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

#Initializing list
sensor_pos = []
sensor_angle = []
magnet_pos = []
magnet_angle = []

#Initializing first sensor with no angle, at point [0,0,0]
sensor_pos.append(np.array([0,0,0]))
sensor_angle.append(np.array([0,0,0]))
ax.scatter(sensor_pos[0][0],sensor_pos[0][1],sensor_pos[0][2],color='black')
for i in range(0,segment_amount):

    #Calculate magnet position starting at origin
    pos = bend(segment_length,[0,0,0],bend_angle[i],bend_direction[i])    
    
    #Apply rotation based on sensor_angle
    pos = angle(pos,sensor_angle[i])
  
    #Move point from starting at origin to starting at sensor position
    pos = pos + sensor_pos[i]

    magnet_pos.append(pos) 

    #Calculate angle of magnet based on other angles and current bend
    magnet_angle.append(np.array([bend_direction[i],bend_angle[i],0]))

    #The next sensor angle is the same as the current stage magnet angle
    sensor_angle.append(sensor_angle[i]+magnet_angle[i])
    
    #Sensor position at last magnet place, for the moment
    sensor_pos.append(magnet_pos[i])

    ax.scatter(magnet_pos[i][0],magnet_pos[i][1],magnet_pos[i][2],color='darkorange')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
