import magpylib as magpy
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box
import magpylib as magpy
from magpylib.source.magnet import Box
from magpylib import source, Collection

s1 = Box(mag=[-575.4,0,-575.4],dim=[6.35,6.35,6.35],pos=[-.167778,26.459675,-82.575249],angle=91.48020981,axis=[.15262477,.97647626,.15231479])

c = magpy.Collection(s1)

print(c.getB([0,0,0]))
