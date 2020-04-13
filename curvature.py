import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#Distance between hall effect & magnet when straight above eachother
segment_length = 300 #mm

#Where the hall effect sensor is in space
mount_pos = [0,0,0]

#How much bend is desired
bend_angle = np.pi/4 #0-pi/2 is suggested

#The direction in which the bend is applied
bend_direction = 0 #0-2pi

#Radius of the circle the segment bend follows
circle_radius = segment_length/bend_angle

#Circle = (x-mount_pos[0])**2 + (y-mount_pos[1])**2 = circle_radius**2

#x = np.cos(bend_direction)*circle_radius + mount_pos[0]
#y = np.sin(bend_direction)*circle_radius + mount_pos[1]

x = np.cos(np.pi-bend_angle)*circle_radius + mount_pos[0]+circle_radius
z = np.sin(np.pi-bend_angle)*circle_radius + mount_pos[2]

print(x)
print(z)

x = x + np.cos(bend_direction)*circle_radius+mount_pos[0]
y = np.sin(bend_direction)*circle_radius+mount_pos[1]

print(x)
print(y)
print(z)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(mount_pos[0],mount_pos[1],mount_pos[2])
ax.scatter(x,y,z)
plt.show()
