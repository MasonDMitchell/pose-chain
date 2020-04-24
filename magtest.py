from chain import Chain
import math
from mpl_toolkits.mplot3d import Axes3D
from random import random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd
from magpylib.source.magnet import Cylinder,Box
import magpylib as magpy

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

segment_amount = 5

x = Chain(segment_amount)
x.update_chain()

sensors = np.transpose(x.sensors_pose)
magnets = np.transpose(x.magnets_pose)

for i in range(segment_amount):
    x.bend_segment(i,np.pi/6,np.pi/3)
x.update_chain()
sensors = np.transpose(x.sensors_pose)
magnets = np.transpose(x.magnets_pose)
rotvecs = np.transpose(x.rotvecs) * x.module_length

lines = x.bend_lines()
for i in range(segment_amount):
    ax.plot(lines[i][0],lines[i][1],lines[i][2],color='black')

#ax.scatter(sensors[0],sensors[1],sensors[2],depthshade=0,edgecolors='black',color='red')
#ax.scatter(magnets[0],magnets[1],magnets[2],depthshade=0,edgecolors='black',color='darkorange')
ax.quiver(magnets[0],magnets[1],magnets[2],rotvecs[0][1:],rotvecs[1][1:],rotvecs[2][1:],color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0,1200)
ax.set_ylim(0,1200)
ax.set_zlim(0,1200)

#Magnet parameters

M = [0,0,575.4]#magnetizaiton
D = [6.35,6.35,6.35]#dimesion

mag = []
sen = []
sen.append(magpy.Sensor(pos=[sensors[0][0],sensors[1][0],sensors[2][0]]))
for i in range(segment_amount):
    rotation_vector = np.cross([0,0,1],[rotvecs[0][i+1],rotvecs[1][i+1],rotvecs[2][i+1]])

    angle = (180*np.arccos(np.dot([0,0,1],[rotvecs[0][i+1],rotvecs[1][i+1],rotvecs[2][i+1]])/(np.linalg.norm([0,0,1])*np.linalg.norm([rotvecs[0][i+1],rotvecs[1][i+1],rotvecs[2][i+1]]))))/math.pi

    mag.append(Box(mag=M, dim=D, pos = [magnets[0][i],magnets[1][i],magnets[2][i]],angle=angle,axis=rotation_vector))
    if(i!=segment_amount-1):
        sen.append(magpy.Sensor(pos = [sensors[0][i+1],sensors[1][i+1],sensors[2][i+1]],angle = angle, axis = rotation_vector))

c = magpy.Collection(mag)

magpy.displaySystem(c,sensors=sen,subplotAx=ax,suppress=True)

#print(sen[6].getB(c))

plt.show()
