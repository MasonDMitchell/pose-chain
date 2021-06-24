import magpylib as magpy
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem
import numpy as np
from tools import createChain, noise
from straight_filter import Filter
import matplotlib.pyplot as plt
def angle_axis2(orientation):
        rotvecs = []
        for i in range(len(orientation)):
            #Generate rotvecs and angles for rotation
            rotvecs.append(orientation[i].as_rotvec())

        rotvecs = np.array(rotvecs)

        angles = np.linalg.norm(rotvecs,axis=2)

        #Get indices that need no rotation
        zero = np.where(angles == 0.0)[0]
        #Replace rotvecs that need no rotation
        rotvecs[zero] = [1,0,0]

        #Ensure rotvecs have length 1
        rotvecs = np.divide(rotvecs,np.reshape(np.repeat(np.linalg.norm(rotvecs,axis=2),3),(1,1,3)))

        #Change radians to degrees for processing
        angles = np.degrees(angles)

        return angles, rotvecs

magnet_array = np.arange(1,2,1)
bend_angle = 0
offset = 0
bend_angle = (bend_angle/180)*np.pi
segment_length = 13.3
print(segment_length)
chain = createChain(1,1,bend_angle,0,segment_length,0)
points = chain.GetPoints(magnet_array)
orient = chain.GetOrientations(magnet_array)
ang1,axis1 = angle_axis2(orient)
points1 = points[0][0]

s1 = Box(mag=(-1320,0,0),dim=(6.35,6.35,6.35),pos=points1,angle=ang1[0],axis=axis1[0][0])
print(s1.getB([0,0,0]))
