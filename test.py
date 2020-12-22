import magpylib as magpy
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem

#s1 = Box(mag=(-151,0,0),dim=(6.35,6.35,6.35),pos=(14,0,0))
s1 = Box(mag=(-575.4,0,0),dim=(6.35,6.35,6.35),pos=(1,0,0))
c = Collection(s1)
print(c.getB([0,0,0]))
magpy.displaySystem(c,direc=True)
