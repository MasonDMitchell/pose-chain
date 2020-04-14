from segment import Segment
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import math

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

line_x=[]
line_y=[]
line_z=[]

mag_x = []
mag_y = []
mag_z = []

rotvec = [1,0,0]

rotvec = rotvec / np.linalg.norm(rotvec)
ax.quiver(0,0,0,200*rotvec[0],200*rotvec[1],200*rotvec[2],color='red')

for i in np.arange(0,2*np.pi/4,.25):
    for j in np.arange(2,20,.75):

        seg = Segment()
        seg.bend_angle = math.pi/j
        seg.bend_direction = i
        seg.apply_rotvec(rotvec)

        mag_x.append(seg.magnet_pose[0])
        mag_y.append(seg.magnet_pose[1])
        mag_z.append(seg.magnet_pose[2])
        
        vec_x = seg.magnet_pose[0] + 30*seg.final_rotvec[0]
        vec_y = seg.magnet_pose[1] + 30*seg.final_rotvec[1]
        vec_z = seg.magnet_pose[2] + 30*seg.final_rotvec[2]

        ax.quiver(seg.magnet_pose[0],seg.magnet_pose[1],seg.magnet_pose[2],50*seg.final_rotvec[0],50*seg.final_rotvec[1],50*seg.final_rotvec[2],color='blue')

        for k in np.arange(1,10,.1):
            seg.bend_angle = math.pi/j/k
            seg.segment_length = 300/k
            seg.apply_rotvec(rotvec)
    
            line_x.append(seg.magnet_pose[0])
            line_y.append(seg.magnet_pose[1])
            line_z.append(seg.magnet_pose[2])

        
        ax.plot(line_x,line_y,line_z,color='black')
        line_x=[]
        line_y=[]
        line_z=[]

ax.scatter(mag_x,mag_y,mag_z,depthshade=0,edgecolors='black',color='darkorange')
ax.scatter(0,0,0,depthshade=0,edgecolors='black',color='darkred')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
ax.set_zlim(0,400)

plt.show()
