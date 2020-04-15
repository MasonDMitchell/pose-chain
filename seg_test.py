from segment import Segment
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sci
import pandas as pd

plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

pi = np.pi

seg = Segment()
seg.bend_angle = pi/2
seg.bend_direction = pi/2

rotvec = [1,0,0]

#seg.apply_rotvec(rotvec)
#ax.scatter(seg.magnet_pose[0],seg.magnet_pose[1],seg.magnet_pose[2])
seg.apply_rotvec(rotvec)
ax.scatter(seg.magnet_pose[0],seg.magnet_pose[1],seg.magnet_pose[2])
print(seg.magnet_pose)
x = seg.bend_line(rotvec)
ax.scatter(x[0],x[1],x[2])

plt.show()
