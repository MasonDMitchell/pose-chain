from chain import Chain
from mpl_toolkits.mplot3d import Axes3D
from random import random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd
import magpylib as magpy

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

segment_amount = 5

chain = Chain(segment_amount)

for i in range(segment_amount):
    chain.bend_segment(i,np.pi/25,np.pi/3)
chain.update_chain()
chain.bend_lines()

for i in range(segment_amount):
    ax.plot(chain.lines[i][0],chain.lines[i][1],chain.lines[i][2],color='black')

for i in np.arange(0,segment_amount,1):
    ax.quiver(chain.segments[i].magnet_pose[0],chain.segments[i].magnet_pose[1],chain.segments[i].magnet_pose[2],chain.module_length*chain.segments[i].final_rotvec[0],chain.module_length*chain.segments[i].final_rotvec[1],chain.module_length*chain.segments[i].final_rotvec[2],color='blue')

mag = []
sen = []
for i in range(segment_amount):
    mag.append(chain.segments[i].magnet)
    sen.append(chain.segments[i].sensor)
c = magpy.Collection(mag)
markerPos = [(0,0,0,'origin')]
magpy.displaySystem(c,subplotAx=ax,suppress=True,markers=markerPos)

print(sen[0].getB(c))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.set_xlim(-0,1200)
#ax.set_ylim(0,1200)
#ax.set_zlim(0,1200)
plt.show()
