from chain import Chain
from mpl_toolkits.mplot3d import Axes3D
from random import random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd
import magpylib as magpy

#Init matplotlib stuff
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

#Amount of segments in chain
segment_amount = 2

#Initialize class with segment amount
chain = Chain(segment_amount)

#Set bends for each segment 
chain.bend_segment(0,np.pi/2,np.pi/4)
chain.bend_segment(1,np.pi/2,0)
#Update chain, always do after changing bend segments
chain.update_chain()
#Calculate lines from sensor to magnets for visualization
chain.bend_lines()

#Plot lines from sensor to magnets
for i in range(segment_amount):
    ax.plot(chain.lines[i][0],chain.lines[i][1],chain.lines[i][2],color='black')

#Plot modules as vectors for each segment
for i in np.arange(0,segment_amount,1):
    temp_mag_pose = chain.segments[i].magnet_pose
    temp_rotvec = chain.module_length * np.array(chain.segments[i].final_rotvec)
    ax.quiver(temp_mag_pose[0],temp_mag_pose[1],temp_mag_pose[2],temp_rotvec[0],temp_rotvec[1],temp_rotvec[2],color='lightblue')

#Create collection for magnets and sensors
mag = []
sen = []
for i in range(segment_amount):
    mag.append(chain.segments[i].magnet)
    sen.append(chain.segments[i].sensor)
c = magpy.Collection(mag)
markerPos = [(0,0,0,'origin')]
#sen[0].getB(c)

#print(mag[1].angle)
#mag[1].rotate(-135,chain.segments[1].final_rotvec)
#print(mag[1].angle)

#Plot magnets and show plot
magpy.displaySystem(c,subplotAx=ax,suppress=True,markers=markerPos)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0,400)
ax.set_ylim(0,400)
ax.set_zlim(0,400)
plt.show()
