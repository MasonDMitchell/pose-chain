import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tools import bend
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

#Purpose of file is to demonstrate where cylinder can bend and the path it takes

x=[]
y=[]
z=[]
#initial pose
sensor = [0,0,0]

#calculating end points for positions
for k in np.arange(0,2*np.pi,.4):
    #calculating places in direction k with bend 0 to pi/2
    for i in np.arange(1.5,50,.4):
        a= bend(300,sensor,np.pi/i,k)
        x.append(a[0])
        y.append(a[1])
        z.append(a[2])
    ax.scatter(x,y,z,depthshade=0,edgecolors='black',color='darkorange')
    
    #Calculating intermediate values to show path of cylinder
    for j in np.arange(1.5,50,.4):
        x=[]
        y=[]
        z=[]
        for i in np.arange(1,10,.1):
            a = bend(300/i,sensor,np.pi/j/i,k)
            x.append(a[0])
            y.append(a[1])
            z.append(a[2])
        x.append(sensor[0])
        y.append(sensor[1])
        z.append(sensor[2])
        ax.plot(x,y,z,color='black')

ax.scatter(sensor[0],sensor[1],sensor[2],depthshade=0,edgecolors='black',color='darkred')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-200,200)
ax.set_ylim(-200,200)
ax.set_zlim(0,400)
plt.show()
