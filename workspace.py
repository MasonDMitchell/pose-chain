import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sci
import pandas as pd
import pickle
plt.style.use('ggplot')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

all_points = pickle.load(open("data/test.p",'rb'))

x = []
y = []
z = []
print(all_points[0][-1][0])
for i in range(len(all_points)):
    x.append(all_points[i][-1][0])
    y.append(all_points[i][-1][1])
    print(all_points[i][-1][2])
    z.append(all_points[i][-1][2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.scatter(x,y,z)
plt.show()
