import pickle
import numpy as np
import scipy
import numpy
import magpylib as magpy
from magpylib.source.magnet import Box
from magpylib import source, Collection
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

data = pickle.load( open( "data/test.p", "rb" ) )

mag = [0,0,-575.4]
dim = [6.35,6.35,6.35]
sen = []

loops = 150

for i in range(5,loops):
    b = Box(mag=mag,dim=dim,pos=data[i][0],angle=data[i][1],axis=data[i][2])
    sen.append(magpy.Sensor(pos=data[i][0],angle=data[i][1],axis=data[i][2]))
    col = Collection(b)
    if(i == (loops-1)):
        h = 1
        magpy.displaySystem(col,sensors=sen)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

length = 3
for i in range(5,loops):
    x = [1,0,0]
    y = [0,1,0]
    z = [0,0,1]

    pos = data[i][0]
    axis = data[i][2]
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(data[i][1])
    r = R.from_rotvec(axis*angle)
    x_rot = r.apply(x)*20
    y_rot = r.apply(y)*20
    z_rot = r.apply(z)*20
    ax.quiver(pos[0],pos[1],pos[2],x_rot[0],x_rot[1],x_rot[2],color='red')
    ax.quiver(pos[0],pos[1],pos[2],y_rot[0],y_rot[1],y_rot[2],color='green')
    ax.quiver(pos[0],pos[1],pos[2],z_rot[0],z_rot[1],z_rot[2],color='blue')
ax.set_xlim(-80,80)
ax.set_ylim(-80,80)
ax.set_zlim(-80,80)

plt.show()
