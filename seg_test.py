from segment import Segment

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sci
import pandas as pd
import magpylib as magpy
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

pi = np.pi

seg = Segment()

seg.bend_angle = pi/2
seg.bend_direction = 3*pi

rotvec = [0,0,1]

seg.apply_rotvec(rotvec,[1,0,0])


x = seg.bend_line(rotvec)
ax.plot(x[0],x[1],x[2],color='black')

c = magpy.Collection(seg.magnet)
magpy.displaySystem(c,subplotAx=ax,suppress=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
'''
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
ax.set_zlim(-200,200)
'''
plt.show()
