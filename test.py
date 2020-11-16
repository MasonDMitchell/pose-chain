import magpylib as magpy
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem

s1 = Box(mag=(-575.4,0,0),dim=(6.35,6.35,6.35),pos=(30,0,0))
c = Collection(s1)
print(c.getB([0,0,0]))
