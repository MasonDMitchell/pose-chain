import magpylib as magpy
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem
import numpy as np
from tools import createChain, noise
from straight_filter import Filter
#s1 = Box(mag=(-151,0,0),dim=(6.35,6.35,6.35),pos=(14,0,0))
#s1 = Box(mag=(-575.4,0,0),dim=(6.35,6.35,6.35),pos=(14,0,0))
#c = Collection(s1)
#print(s1.getB([0,0,0]))


#magpy.displaySystem(c,direc=True)
bend_angle = np.radians(80)

chain = createChain(particles=1,
                        segments=1,
                        bend_angle=bend_angle,
                        bend_direction=np.pi,
                        bend_length=14,
                        straight_length=0)

x = Filter(chain,noise)
x.compute_flux()
print(x.Bv)
