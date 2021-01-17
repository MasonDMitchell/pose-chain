from tools import createChain,noise
import numpy as np
from straight_filter import Filter
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem

chain = createChain(1,np.pi/2,0,10)

x = Filter(chain,noise)

magnet_array = np.array([0,1])
sensor_array = np.array([.5])

print(x.compute_flux())
print("MAGNET POS")
print(x.chain.GetPoints(magnet_array))
mag_orient = x.chain.GetOrientations(magnet_array)
mag1 = mag_orient[0].as_rotvec()
mag2 = mag_orient[1].as_rotvec()
print(mag1)
print(mag2)
print(x.angle_axis2(mag_orient))
print("SENSOR POS")
print(x.chain.GetPoints(sensor_array))
sensor_orient = x.chain.GetOrientations(sensor_array)
sensor = sensor_orient[0].as_rotvec()
print(sensor)
print(x.angle_axis2(sensor_orient))

s1 = Box(mag=(-575.4,0,0),dim=(6.35,6.35,6.35),pos=(0,0,0),angle=0,axis=(1,0,0))
s2 = Box(mag=(-575.4,0,0),dim=(6.35,6.35,6.35),pos=(0,12.7323954,0),angle=180,axis=(0,0,1))

sens = magpy.Sensor(pos=(6.36619772,6.36619772,0),angle=90,axis=(0,0,1))
c = Collection(s1,s2)
print(c.getB(pos=(6.36619772,6.36619772,0)))
print(sens.getB(c))
