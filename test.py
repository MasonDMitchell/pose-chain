from segment import Segment
import numpy as np
import scipy as sci
import math

x = Segment()
x.bend_angle = math.pi/2
x.bend_direction = math.pi/2
print(x.magnet_pose)
x.update_bend()
print(x.magnet_pose)
print(x.final_rotvec)
