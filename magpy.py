import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import pandas as pd
from magpylib.source.magnet import Cylinder, Box
import magpylib as magpy

#Can probably be made in a list then addressed from there
mag = Box(mag=[500,0,500],dim=[5,5,5])
mag2 = Box(mag=[500,0,500],dim=[2,2,2])

#Setting magnet positions
mag.setPosition([0,0,10])

#Can declare magnets in lists
mag_col = magpy.Collection([mag,mag2])

#Creating sensors & setting pos in same command
se1 = magpy.Sensor(pos=[10,0,0])

#Animated ?
magpy.displaySystem(mag_col,sensors=[se1])

print(mag.axis)
