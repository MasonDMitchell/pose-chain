import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tools import bend
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

a=[]
b=[]
c=[]
sensor = [0,0,0]
for k in np.arange(0,2*np.pi,.4):
    for i in np.arange(1.5,50,.4):
        x= bend(300,sensor,np.pi/i,k)
        a.append(x[0])
        b.append(x[1])
        c.append(x[2])
    ax.scatter(a,b,c,depthshade=0,edgecolors='black',color='darkorange')
    for j in np.arange(1.5,50,.4):
        a=[]
        b=[]
        c=[]
        for i in np.arange(1,10,.1):
            x = bend(300/i,sensor,np.pi/j/i,k)
            a.append(x[0])
            b.append(x[1])
            c.append(x[2])
        a.append(sensor[0])
        b.append(sensor[1])
        c.append(sensor[2])
        ax.plot(a,b,c,color='black')

ax.scatter(sensor[0],sensor[1],sensor[2],depthshade=0,edgecolors='black',color='darkred')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
ax.set_zlim(0,400)
plt.show()
