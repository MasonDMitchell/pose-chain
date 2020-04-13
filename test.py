import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sci
from scipy.spatial.transform import Rotation as R
from tools import angle,bend
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

pi = np.pi
sensor = [0,0,0]
sensor1 = bend(300,sensor,(pi/2),0)
for i in np.arange(1,8,.2):
    pos = bend(300/i,sensor,(pi/2)/i,pi/2)
    pos = angle(pos,[0,pi/2,0])
    pos = angle(pos,[pi/2,0,0])
    ax.scatter(pos[0],pos[1],pos[2],color='black')

ax.scatter(sensor[0],sensor[1],sensor[2])
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
ax.set_zlim(-200,200)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

