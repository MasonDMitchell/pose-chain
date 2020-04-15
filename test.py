from chain import Chain
from mpl_toolkits.mplot3d import Axes3D
from random import random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

segment_amount = 100

x = Chain(segment_amount)
x.update_chain()

sensors = np.transpose(x.sensors_pose)
magnets = np.transpose(x.magnets_pose)
#ax.scatter(sensors[0],sensors[1],sensors[2],depthshade=0,edgecolors='black',color='darkred')
#ax.scatter(magnets[0],magnets[1],magnets[2],depthshade=0,edgecolors='black',color='darkorange')
#plt.show()


for i in range(segment_amount):
    x.bend_segment(i,np.pi/6,np.pi/3)
x.update_chain()
sensors = np.transpose(x.sensors_pose)
magnets = np.transpose(x.magnets_pose)
rotvecs = np.transpose(x.rotvecs) * x.module_length

lines = x.bend_lines()
for i in range(segment_amount):
    ax.plot(lines[i][0],lines[i][1],lines[i][2],color='black')

ax.scatter(sensors[0],sensors[1],sensors[2],depthshade=0,edgecolors='black',color='red')
ax.scatter(magnets[0],magnets[1],magnets[2],depthshade=0,edgecolors='black',color='darkorange')
ax.quiver(magnets[0],magnets[1],magnets[2],rotvecs[0][1:],rotvecs[1][1:],rotvecs[2][1:],color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.set_xlim(-0,1200)
#ax.set_ylim(0,1200)
#ax.set_zlim(0,1200)
plt.show()
