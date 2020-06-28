import magpylib as magpy
from magpylib.source.magnet import Box

s1 = Box(mag=[0,0,-575.4],dim=[6.35,6.35,6.35],pos=[34.684441,0,0])

c = magpy.Collection(s1)

print(c.getB([0,0,0]))
