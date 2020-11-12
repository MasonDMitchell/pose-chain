import numpy as np
import matplotlib.pyplot as plt
from chain import CompositeSegment
from segment import ConstLineSegment, CircleSegment
from tools import createChain
def spiral_params():
    #Spiral Parameters
    a = 0
    b = .02
    c = 1

    #Create spiral
    t = np.arange(0,10*np.pi,.01)
    x = (a + b*t) * np.cos(t)
    y = (a + b*t) * np.sin(t)

    #Attain alpha & beta values from x,y values
    vector = np.array([x,y])
    alpha = np.linalg.norm(vector,axis=0)
    beta = np.arctan2(vector[1],vector[0])

